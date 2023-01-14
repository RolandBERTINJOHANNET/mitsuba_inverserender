#include "../../ext/drjit/include/drjit/autodiff.h"
#include <iostream>
#include <mitsuba/core/argparser.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/jit.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/xml.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/scene.h>
#include <map>

#include <drjit/autodiff.h>
#include <drjit-core/containers.h>
#include "texbm.h"

#include <fstream>

#if !defined(_WIN32)
#include <signal.h>
#endif

using namespace mitsuba;

static void help(int thread_count) {
    std::cout << util::info_build(thread_count) << std::endl;
    std::cout << util::info_copyright() << std::endl;
    std::cout << util::info_features() << std::endl;
    std::cout << R"(
Usage: mitsuba [options] <One or more scene XML files>

Options:

    -h, --help
        Display this help text.

    -m, --mode
        Request a specific mode/variant of the renderer

        Default: )" MI_DEFAULT_VARIANT R"(

        Available:
              )"
              << string::indent(MI_VARIANTS, 14) << R"(
    -v, --verbose
        Be more verbose. (can be specified multiple times)

    -t <count>, --threads <count>
        Render with the specified number of threads.

    -D <key>=<value>, --define <key>=<value>
        Define a constant that can referenced as "$key" within the scene
        description.

    -s <index>, --sensor <index>
        Index of the sensor to render with (following the declaration order
        in the scene file). Default value: 0.

    -u, --update
        When specified, Mitsuba will update the scene's XML description
        to the latest version.

    -a <path1>;<path2>;.., --append <path1>;<path2>
        Add one or more entries to the resource search path.

    -o <filename>, --output <filename>
        Write the output image to the file "filename".

 === The following options are only relevant for JIT (CUDA/LLVM) modes ===

    -O [0-5]
        Enables successive optimizations (default: -O5):
          (0. all disabled, 1: de-duplicate virtual functions,
           2: constant propagation, 3. value numbering,
           4. virtual call optimizations, 5. loop optimizations)

    -S
        Dump the PTX or LLVM intermediate representation to the console

    -W
        Instead of compiling a megakernel, perform rendering using a
        series of wavefronts. Specify twice to unroll both loops *and*
        virtual function calls.

    -V <width>
        Override the vector width of the LLVM backend ('width' must be
        a power of two). Values of 4/8/16 cause SSE/NEON, AVX, or AVX512
        registers being used (if supported). Going beyond the natively
        supported width is legal and causes arithmetic operations to be
        replicated multiple times.

    -P
        Force parallel scene loading, which is disabled by default
        in JIT modes since interferes with the ability to reuse
        cached compiled kernels across separate runs of the renderer.

)";
}

float LR = .5;
int NB_ITER = 400;


//----------------------------------------------------------CUSTOM RENDER OP----------------------------------------------
using Scn = Scene<dr::DiffArray<dr::CUDAArray<float>>,
                  Color<dr::DiffArray<dr::CUDAArray<float>>, 3>> *;
using Dif = dr::DiffArray<dr::CUDAArray<float>>;

class RenderOp
    : public dr::CustomOp<dr::DiffArray<dr::CUDAArray<float>>,
                          dr::DiffArray<dr::CUDAArray<float>>,Scn,
                                     uint32_t /*sensor index*/, int /* seed */, int /* spp */, bool /* develop */,
                                     bool /* evaluate */,
                                     Dif /*param*/> {

public:
    /**
     * Evaluate the custom function in primal mode. The inputs will be detached
     * from the AD graph, and the output *must* also be detached.
     */
    dr::DiffArray<dr::CUDAArray<float>> eval(const Scn &scene,
                                             const uint32_t& sensor,
                                            const int& seed, const int& spp, const bool& develop,
                                            const bool& evaluate,
                                            const Dif &d1) override {

        //set params for the gradient-render (in backward).
        this->scene = scene;
        this->sensor = sensor;
        this->spp    = spp;
        this->seed   = sample_tea_32<uint32_t>(seed, 1).first;
        this->develop = develop;
        this->evaluate = evaluate;
        this->idcs     = idcs;



        {
            dr::suspend_grad<dr::DiffArray<dr::CUDAArray<float>>> sp(true);

            auto integrator = scene->integrator();
            auto rendered = integrator->render(
                scene, (uint32_t) sensor, seed /* seed */, spp /* spp */,
                develop /* develop */, evaluate /* evaluate */);
            return rendered.array();
        }
    }

    /// Callback to implement forward-mode derivatives
    void forward() override { 
        std::cout << "forward : " << name()<< std::endl;
    }

    /// Callback to implement backward-mode derivatives
    void backward() override {
        //re-put implicit params
        /* for (uint32_t idx : this->idcs) {
            m_implicit_in.push_back(idx);
        }*/
        //std::cout << "backward" << std::endl;
        jit_set_flag(JitFlag::LoopRecord, false);

        //render an image
        auto integrator = scene->integrator();
        auto rendered = integrator->render(
            scene, (uint32_t) sensor, seed /* seed */, spp /* spp */,
            develop /* develop */, evaluate /* evaluate */).array();

        //go back
        dr::backward_from(rendered * this->grad_out());
    }

    /// Return a descriptive name (used in GraphViz output)
    const char *name() const override {
        return "custom render operation";
    }

    private:
    Scn scene;
    uint32_t sensor;
    int seed;
    int spp;
    bool develop;
    bool evaluate;
    Color<dr::DiffArray<dr::CUDAArray<float>>> idcs;
};


//-----------------------------------------------------------------------sfm data parser---------------

float getFloatDiff(dr::DiffArray<dr::CUDAArray<float>> val) {
    return val.entry(0);
}

void rotationMatrixToQuaternion(
    mitsuba::Transform<
        mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 4>> rotation,
        std::vector<float> &q) {
    q.resize(4);
    float trace = getFloatDiff(rotation.matrix.data()[0][0]) +
                  getFloatDiff(rotation.matrix.data()[1][1]) +
                  getFloatDiff(rotation.matrix.data()[2][2]);
    if (trace > 0) {       // I changed M_EPSILON to 0
        float s = 0.5f / sqrtf(trace + 1.0f);
        q[3]    = 0.25f / s;
        q[0] = (getFloatDiff(rotation.matrix.data()[2][1]) -
               getFloatDiff(rotation.matrix.data()[1][2])) *
              s;
        q[1] = (getFloatDiff(rotation.matrix.data()[0][2]) -
               getFloatDiff(rotation.matrix.data()[2][0])) *
              s;
        q[2] = (getFloatDiff(rotation.matrix.data()[1][0]) -
               getFloatDiff(rotation.matrix.data()[0][1])) *
              s;
    } else {
        if (getFloatDiff(rotation.matrix.data()[0][0]) >
                getFloatDiff(rotation.matrix.data()[1][1]) &&
            getFloatDiff(rotation.matrix.data()[0][0]) >
                getFloatDiff(rotation.matrix.data()[2][2])) {
            float s =
                2.0f * sqrtf(1.0f + getFloatDiff(rotation.matrix.data()[0][0]) -
                             getFloatDiff(rotation.matrix.data()[1][1]) -
                             getFloatDiff(rotation.matrix.data()[2][2]));
            q[3] = (getFloatDiff(rotation.matrix.data()[2][1]) -
                   getFloatDiff(rotation.matrix.data()[1][2])) /
                  s;
            q[0]     = 0.25f * s;
            q[1] = (getFloatDiff(rotation.matrix.data()[0][1]) +
                   getFloatDiff(rotation.matrix.data()[1][0])) /
                  s;
            q[2] = (getFloatDiff(rotation.matrix.data()[0][2]) +
                   getFloatDiff(rotation.matrix.data()[2][0])) /
                  s;
        } else if (getFloatDiff(rotation.matrix.data()[1][1]) >
                   getFloatDiff(rotation.matrix.data()[2][2])) {
            float s =
                2.0f * sqrtf(1.0f + getFloatDiff(rotation.matrix.data()[1][1]) -
                             getFloatDiff(rotation.matrix.data()[0][0]) -
                             getFloatDiff(rotation.matrix.data()[2][2]));
            q[3] = (getFloatDiff(rotation.matrix.data()[0][2]) -
                   getFloatDiff(rotation.matrix.data()[2][0])) /
                  s;
            q[0] = (getFloatDiff(rotation.matrix.data()[0][1]) +
                   getFloatDiff(rotation.matrix.data()[1][0])) /
                  s;
            q[1]     = 0.25f * s;
            q[2] = (getFloatDiff(rotation.matrix.data()[1][2]) +
                   getFloatDiff(rotation.matrix.data()[2][1])) /
                  s;
        } else {
            float s =
                2.0f * sqrtf(1.0f + getFloatDiff(rotation.matrix.data()[2][2]) -
                             getFloatDiff(rotation.matrix.data()[0][0]) -
                             getFloatDiff(rotation.matrix.data()[1][1]));
            q[3] = (getFloatDiff(rotation.matrix.data()[1][0]) -
                   getFloatDiff(rotation.matrix.data()[0][1])) /
                  s;
            q[0] = (getFloatDiff(rotation.matrix.data()[0][2]) +
                   getFloatDiff(rotation.matrix.data()[2][0])) /
                  s;
            q[1] = (getFloatDiff(rotation.matrix.data()[1][2]) +
                   getFloatDiff(rotation.matrix.data()[2][1])) /
                  s;
            q[2]     = 0.25f * s;
        }
    }
}

void parse_sfm(
        std::string path,
        std::vector<mitsuba::Transform<
        mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 4>>>& res_transforms,
        std::vector<std::string>& filenames) {
    std::ifstream f; f.open(path);
    std::string line;

    while (std::getline(f,line)) {
        // store filename
        filenames.push_back(line);
        // initialize the view matrix
        mitsuba::Transform<
            mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 4>>
            test;

        // add rotation
        for (int i = 0; i < 9; i++) {
            std::getline(f, line, (i == 8 ? '\n' : ' '));
            test.matrix.data()[i % 3][i / 3] = atof(line.c_str());
        }
        // add translation
        std::vector<float> translation;
        for (int i = 0; i < 3; i++) {
            std::getline(f, line, (i == 2 ? '\n' : ' '));
            test.matrix.data()[3][i] = atof(line.c_str());
            translation.push_back(atof(line.c_str()));
        }

        mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 3> eye(
            translation[0], translation[1], translation[2]);
        mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 3> target(
            0, 0., 0.);
        mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 3> up(0, 1,
                                                                        0);
        test = mitsuba::Transform<mitsuba::Point<
            drjit::DiffArray<drjit::CUDAArray<float>>, 4>>::look_at(eye, target,
                                                                    up);
        std::cout << "the matrix : " << test << std::endl;

        /*// get rotation
        mitsuba::Transform<
            mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 4>>
            rotation;
        for (int i = 0; i < 9; i++) {
            std::getline(f, line, (i == 8 ? '\n' : ' '));
            rotation.matrix.data()[i % 3][i / 3] = atof(line.c_str());
        }
        std::vector<float> q;
        rotationMatrixToQuaternion(rotation, q);
        Vector<dr::DiffArray<dr::CUDAArray<float>>, 3> axis(q[0], q[1], q[2]);
        test.rotate(axis, q[3]);
        // add translation
        for (int i = 0; i < 3; i++) {
            std::getline(f, line, (i == 2 ? '\n' : ' '));
            test.matrix.data()[3][i] = atof(line.c_str());
        }*/

        res_transforms.push_back(test);
    }
}



template<typename T, typename Q>
void add_if_in_map(std::map<T, Q>& map, Q val,T key) {
    bool found = false;
    for (auto elt : map) {
        if (elt.second == val) {
            found = true;
        }
    }
    if (!found) {
        //std::cout << "added element " << key << std::endl;
        map[key] = val;
    } else {
        //std::cout << "did not add element " << key << std::endl;
    }
}

//-------------------------------------------------------------------------------CUSTOM
class CustomTraversalCallback : public TraversalCallback {
public:
    std::string paramName = std::string("");
    std::map<std::string,void*> paramMap;
    int i = 0;

    CustomTraversalCallback() {}
    void put_parameter_impl(const std::string &name, void *ptr, uint32_t flags,
                            const std::type_info &type) {


        add_if_in_map(paramMap, ptr, paramName+ std::string(".") + name);

    }
    void put_object(const std::string &name, Object *obj, uint32_t flags) {
        CustomTraversalCallback cb;
        
        cb.paramName = paramName + std::string(".")+name;//add new recursion name
        cb.paramMap = paramMap;//might have been better to just pass the reference.
        (obj)->traverse(&cb);

        if (cb.paramName == ".image") {
            add_if_in_map(paramMap, (void *) obj, std::string(".image"));
        }
        if (paramName == ".TallBox") {
            std::cout << "parameters added : " << i << std::endl;
            i++;
            add_if_in_map(paramMap, (void *) obj, std::string(".TallBox"));
        }

        for (auto val : cb.paramMap) {
            paramMap[val.first] = val.second;
        }
    }
};


//-----------------------------------------------------------OPTIMIZER

enum OPT_TYPE { Momentum, RMSEProp, Adam };

class Optimizer {
private:
    float lr, beta_1, beta_2;
    OPT_TYPE opt_type;
    std::map<std::string, void *> paramMap;
    dr::DiffArray<dr::CUDAArray<float>> velocity;
    dr::DiffArray<dr::CUDAArray<float>> rmse_val;
    int iter;
    float EPSILON = 0.00000008;


public:
    Optimizer(float lr, float beta_1, float beta_2,OPT_TYPE opt_type) {
        this->lr = lr;
        this->beta_1 = beta_1;
        this->beta_2 = beta_2;
        this->opt_type = opt_type;
        velocity       = dr::DiffArray<dr::CUDAArray<float>>();
        rmse_val       = dr::DiffArray<dr::CUDAArray<float>>();
        iter                = 0;
    }

    void traverse(Scn scen) {
        CustomTraversalCallback cb;
        scen->traverse(&cb);
        paramMap = cb.paramMap;
    }

    template <typename T> T *get_param(std::string param, T model) {
        return static_cast<T *>(paramMap[param]);
    }

    void printParams() {
        std::cout << "\n";
        for (auto pr : paramMap) {
            std::cout << pr.first << std::endl;
        }
        std::cout << "\n";
    }

    void step(std::string paramname,
              Color<dr::DiffArray<dr::CUDAArray<float>>, 3> model) {
        // get the param as correct type
        auto param = get_param(paramname, model);
        // update it
        auto param_grad = dr::grad(*param);
        std::cout << "param grad : " << param_grad << std::endl;
            for (int i = 0; i < 3; i++) {
                auto value = (dr::DiffArray<dr::CUDAArray<float>>(
                    dr::detach((*param)[i]) - LR * param_grad[i]));
                //value      = dr::clamp(value, 0., 1.);
                dr::enable_grad(value);
                (*param)[i] = value;
            }
    }

    void step(std::string paramname,
              dr::Tensor<dr::DiffArray<dr::CUDAArray<float>>> model) {
        // get the param as correct type
        auto param = get_param(paramname, model);
        // get the gradient
        auto param_grad = dr::grad((*param).array());
        std::cout << "param grad : " << param_grad << std::endl;

        //prepare the opt values
        accum_opt_params(param_grad);

        //update the parameter
        auto value =
            dr::zeros<dr::DiffArray<dr::CUDAArray<float>>>(
            param->size());
        switch (opt_type) {
            case Momentum:
                value = (dr::DiffArray<dr::CUDAArray<float>>(
                    dr::detach((*param).array()) - LR * velocity));
            case RMSEProp:
                value = (dr::DiffArray<dr::CUDAArray<float>>(
                    dr::detach((*param).array()) -
                    LR * param_grad / (dr::sqrt(rmse_val) + EPSILON)));
            case Adam:
                clear_bias_adam_params();
                value = (dr::DiffArray<dr::CUDAArray<float>>(
                    dr::detach((*param).array()) -
                    LR * velocity / (dr::sqrt(rmse_val) + EPSILON)));
            default:
                value = (dr::DiffArray<dr::CUDAArray<float>>(
                    dr::detach((*param).array()) - LR * param_grad));
        }
        //value      = dr::clamp(value, 0., 1.);

        dr::enable_grad(value);
        (*param).array() = value;
    }

    void accum_opt_params(dr::DiffArray<dr::CUDAArray<float>> gradient) {
        std::cout<<"initializing velocity and rmse : "<<std::endl;
        if (velocity.size() != gradient.size() ||//if the values are not initialized or anything  strange has happened to the
            rmse_val.size() != gradient.size()) {//shapes, you reset everything to 0.
            velocity = dr::zeros<dr::DiffArray<dr::CUDAArray<float>>>(gradient.size());
            rmse_val =
                dr::zeros<dr::DiffArray<dr::CUDAArray<float>>>(gradient.size());
        }
        std::cout << "done initializing velocity and rmse : " << std::endl;
        velocity = (1 - beta_1) * gradient + beta_1 * velocity;
        rmse_val = (1 - beta_2) * dr::sqr(gradient) + beta_2 * rmse_val;
        iter++;
    }
    void clear_bias_adam_params() {
        velocity = velocity / (1 - pow(beta_1, iter));
        rmse_val = rmse_val / (1 - pow(beta_2, iter));
    }

    void param_changed(std::string name) {
        std::cout << "getting bitmap"<<std::endl;
        // notify params changed :
        void *param_model_ = nullptr;
        auto bmp =
            (BitmapTexture<dr::DiffArray<dr::CUDAArray<float>>,
                           Color<dr::DiffArray<dr::CUDAArray<float>>, 3>> *)
                get_param(std::string(name), param_model_);

        std::vector<std::string> keys;
        keys.push_back("data");
        (*bmp).parameters_changed(keys);
    }
};

//-------------------------------------------------------------------------------CUSTOM
//END

std::function<void(void)> develop_callback;
std::mutex develop_callback_mutex;

template <typename Float, typename Spectrum>
void scene_static_accel_initialization() {
    Scene<Float, Spectrum>::static_accel_initialization();
}

template <typename Float, typename Spectrum>
void scene_static_accel_shutdown() {
    Scene<Float, Spectrum>::static_accel_shutdown();
}

template <typename Float, typename Spectrum>
void render(Object *scene_orig_, size_t sensor_i,
            fs::path filename) {
    auto *scene_orig = dynamic_cast<Scene<Float, Spectrum> *>(scene_orig_);
    if (!scene_orig)
        Throw("Root element of the input file must be a <scene> tag!");
    auto film_orig       = scene_orig->sensors()[sensor_i]->film();
    auto integrator_orig = scene_orig->integrator();

    /* critical section */ {
        std::lock_guard<std::mutex> guard(develop_callback_mutex);
        develop_callback = [&]() { film_orig->write(filename); };
    }
    //--------------------------------------------------------------------------------myChanges
    std::cout << "\n\n--------rendering reference image----------"
            << std::endl;

    std::vector<mitsuba::Transform<
        mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 4>>>
        res_transforms;
    std::vector<std::string> image_names;

    parse_sfm("C:/Users/Orlando/Desktop/sfm/sfm_result/reconstruction_global/"
              "sfm_relevant_data.txt",
              res_transforms, image_names);

    if constexpr (std::is_same_v<Float, dr::DiffArray<dr::CUDAArray<float>>>) {
        Optimizer opt(LR, 0.9, 0.999, Adam); // default beta values.
        opt.traverse(scene_orig);
        opt.printParams();

        for (int i = 0; i < res_transforms.size(); i++) {
            std::cout << "in if" << std::endl;
            // change camera matrix to the view transform
            std::string camname = ".PerspectiveCamera.to_world";
            mitsuba::Transform<
                mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 4>>
                cammodel;
            auto *camMat = opt.get_param(camname, cammodel);
            // change its value
            (*camMat).matrix = res_transforms[i].matrix;

            // render image
            auto renderedRef = integrator_orig->render(
                scene_orig, (uint32_t) sensor_i, 0, 1000, true, true);
            auto imageRef = renderedRef.array();

            // save to png file
            {
                auto bm = film_orig->bitmap();

                bm = (*bm).convert(Bitmap::PixelFormat::RGB,
                                   Struct::Type::UInt8, true);
                (*bm).write("C:/Users/Orlando/Desktop/test_mitsuba/images/"
                            "out_dir_wallcolor/weed_bun_"+std::to_string(i)+".png");
            }
        }
    }
        

    
    /*
    
    
    std::cout << "\n\n--------retrieving the tensor----------" << std::endl;
    //std::cout << "type of the array inside of it : " << typeid(rendered.array()).name() << std::endl;
    if constexpr (std::is_same_v<Float,dr::DiffArray<dr::CUDAArray<float>>>)   
    {
        Optimizer opt(LR,0.9,0.999,Adam);//default beta values.

        opt.traverse(scene_orig);
        opt.printParams();


        
        dr::Tensor<dr::DiffArray<dr::CUDAArray<float>>> parammodel;
         std::cout << opt.get_param(".image.data",
                                   parammodel)
                         ->size()
                  << std::endl;
        

        /*
        //get the camera parameter as a transform.
        std::string camname = ".PerspectiveCamera.to_world";
        mitsuba::Transform<
            mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 4>
        > cammodel;
        auto *camMat = opt.get_param(camname, cammodel);
        // change its value
        //mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 3> eye(
        //    0, 2, 7);
        //mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 3> target(0,1,0);
        //mitsuba::Point<drjit::DiffArray<drjit::CUDAArray<float>>, 3> up(0,1,0);
        //(*camMat) = mitsuba::Transform<mitsuba::Point<
        //    drjit::DiffArray<drjit::CUDAArray<float>>, 4>>::look_at(eye,target,up);

        //get the positions parameter as a tensor.
        std::string paramname = ".TallBox.vertex_positions";
        dr::Tensor<dr::DiffArray<dr::CUDAArray<float>>> parammodel;
        dr::Tensor<dr::DiffArray<dr::CUDAArray<float>>> *positions =
            opt.get_param(paramname, parammodel);

        //get the mesh object
        std::string meshname = ".TallBox";
        auto mesh = (*scene_orig).shapes()[1];
        std::cout << "mesh : "
                  << (*mesh).to_string()<<std::endl;


        dr::enable_grad(*positions);
        //gradient descent
        for (int iteration = 0;iteration<NB_ITER;iteration++)
        {
            // render image
            auto image = dr::custom<RenderOp, Scn, uint32_t, int, int, bool,
                                    bool, Dif>(
                    scene_orig, (uint32_t) sensor_i, 0, 40, true, false,
                    (*positions).array());
            // seed spp develop evaluate

            // save to png file
            {
                auto bm = film->bitmap();

                bm = (*bm).convert(Bitmap::PixelFormat::RGB,
                                   Struct::Type::UInt8, true);
                (*bm).write(std::string("C:/Users/Orlando/Desktop/test_mitsuba/images/"
                             "shape/shape_changed/weed_bun_it_") +
                            std::to_string(iteration) + std::string(".png"));
                std::cout << "wrote to : "
                          << std::string(
                                 "C:/Users/Orlando/Desktop/test_mitsuba/images/"
                                 "shape/shape_changed/weed_bun_it_") +
                                 std::to_string(iteration) + std::string(".png")<<std::endl;
            }
            // print loss

            auto loss = dr::mean(dr::sqr(imageRef - image));
            std::cout << "loss : " << loss << std::endl;

            // get gradient of param
            dr::backward(loss, (uint32_t) drjit::ADFlag::Default);

            // update parameter :
            opt.step(paramname, parammodel);
            //opt.param_changed(".image");
            (*scene_orig).parameters_changed();
            (*mesh).parameters_changed();

            std::cout << "param after update : " << (*positions).array() << "\n"
                      << std::endl;
        }
    }
    /* critical section */ {
        std::lock_guard<std::mutex> guard(develop_callback_mutex);
        develop_callback = nullptr;
    }
}

#if !defined(_WIN32)
// Handle the hang-up signal and write a partially rendered image to disk
void hup_signal_handler(int signal) {
    if (signal != SIGHUP)
        return;
    std::lock_guard<std::mutex> guard(develop_callback_mutex);
    if (develop_callback)
        develop_callback();
}
#endif

int main(int argc, char *argv[]) {
    Jit::static_initialization();
    Class::static_initialization();
    Thread::static_initialization();
    Logger::static_initialization();
    Bitmap::static_initialization();

    // Ensure that the mitsuba-render shared library is loaded
    librender_nop();

    ArgParser parser;
    using StringVec   = std::vector<std::string>;
    auto arg_threads  = parser.add(StringVec{ "-t", "--threads" }, true);
    auto arg_verbose  = parser.add(StringVec{ "-v", "--verbose" }, false);
    auto arg_define   = parser.add(StringVec{ "-D", "--define" }, true);
    auto arg_sensor_i = parser.add(StringVec{ "-s", "--sensor" }, true);
    auto arg_output   = parser.add(StringVec{ "-o", "--output" }, true);
    auto arg_update   = parser.add(StringVec{ "-u", "--update" }, false);
    auto arg_help     = parser.add(StringVec{ "-h", "--help" });
    auto arg_mode     = parser.add(StringVec{ "-m", "--mode" }, true);
    auto arg_paths    = parser.add(StringVec{ "-a" }, true);
    auto arg_extra    = parser.add("", true);

    // Specialized flags for the JIT compiler
    auto arg_optim_lev = parser.add(StringVec{ "-O" }, true);
    auto arg_load_par  = parser.add(StringVec{ "-P" });
    auto arg_wavefront = parser.add(StringVec{ "-W" });
    auto arg_source    = parser.add(StringVec{ "-S" });
    auto arg_vec_width = parser.add(StringVec{ "-V" }, true);

    xml::ParameterList params;
    std::string error_msg, mode;

#if !defined(_WIN32)
    /* Initialize signal handlers */
    struct sigaction sa;
    sa.sa_handler = hup_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGHUP, &sa, nullptr))
        Log(Warn, "Could not install a custom signal handler!");
#endif

    try {
        // Parse all command line options
        parser.parse(argc, argv);

#if defined(NDEBUG)
        int log_level = 0;
#else
        int log_level = 1;
#endif
        auto arg = arg_verbose;
        while (arg && *arg) {
            log_level++;
            arg = arg->next();
        }

        // Set the log level
        auto logger                           = Thread::thread()->logger();
        mitsuba::LogLevel log_level_mitsuba[] = { Info, Debug, Trace,Warn };

        logger->set_log_level(log_level_mitsuba[std::min(log_level, 2)]);
        //added line
        logger->set_log_level(log_level_mitsuba[3]);

#if defined(MI_ENABLE_CUDA) || defined(MI_ENABLE_LLVM)
        ::LogLevel log_level_drjit[] = { ::LogLevel::Error, ::LogLevel::Warn,
                                         ::LogLevel::Info,  ::LogLevel::InfoSym,
                                         ::LogLevel::Debug, ::LogLevel::Trace };
        jit_set_log_level_stderr(log_level_drjit[std::min(log_level, 6)]);
        // added line
        jit_set_log_level_stderr(log_level_drjit[0]);
#endif

        // Initialize nanothread with the requested number of threads
        size_t thread_count = Thread::thread_count();
        if (*arg_threads) {
            thread_count = arg_threads->as_int();
            if (thread_count < 1) {
                Log(Warn, "Thread count should be greater than 0. It will be "
                          "set to 1 instead.");
                thread_count = 1;
            }
        }
        Thread::set_thread_count(thread_count);

        while (arg_define && *arg_define) {
            std::string value = arg_define->as_string();
            auto sep          = value.find('=');
            if (sep == std::string::npos)
                Throw("-D/--define: expect key=value pair!");
            params.emplace_back(value.substr(0, sep), value.substr(sep + 1),
                                false);
            arg_define = arg_define->next();
        }
        mode      = (*arg_mode ? arg_mode->as_string() : MI_DEFAULT_VARIANT);
        bool cuda = string::starts_with(mode, "cuda_");
        bool llvm = string::starts_with(mode, "llvm_");

#if defined(MI_ENABLE_CUDA)
        if (cuda)
            jit_init((uint32_t) JitBackend::CUDA);
#endif

#if defined(MI_ENABLE_LLVM)
        if (llvm)
            jit_init((uint32_t) JitBackend::LLVM);
#endif

#if defined(MI_ENABLE_LLVM) || defined(MI_ENABLE_CUDA)
        if (cuda || llvm) {
            if (*arg_optim_lev) {
                int lev = arg_optim_lev->as_int();
                jit_set_flag(JitFlag::VCallDeduplicate, lev > 0);
                jit_set_flag(JitFlag::ConstProp, lev > 1);
                jit_set_flag(JitFlag::ValueNumbering, lev > 2);
                jit_set_flag(JitFlag::VCallOptimize, lev > 3);
                jit_set_flag(JitFlag::LoopOptimize, lev > 4);
            }

            if (*arg_wavefront) {
                jit_set_flag(JitFlag::LoopRecord, false);
                if (arg_wavefront->next())
                    jit_set_flag(JitFlag::VCallRecord, false);
            }

            if (*arg_source)
                jit_set_flag(JitFlag::PrintIR, true);

            if (*arg_vec_width && llvm) {
                uint32_t width = arg_vec_width->as_int();
                if (!math::is_power_of_two(width))
                    Throw("Value specified to the -V argument must be a power "
                          "of two!");

                std::string target_cpu      = jit_llvm_target_cpu(),
                            target_features = jit_llvm_target_features();

                jit_llvm_set_target(target_cpu.c_str(), target_features.c_str(),
                                    (uint32_t) width);
            }
        }
#else
        DRJIT_MARK_USED(arg_wavefront);
        DRJIT_MARK_USED(arg_optim_lev);
        DRJIT_MARK_USED(arg_source);
#endif

        if (!cuda && !llvm &&
            (*arg_optim_lev || *arg_wavefront || *arg_source || *arg_vec_width))
            Throw("Specified an argument that only makes sense in a JIT "
                  "(LLVM/CUDA) mode!");

        Profiler::static_initialization();
        color_management_static_initialization(cuda, llvm);

        MI_INVOKE_VARIANT(mode, scene_static_accel_initialization);

        size_t sensor_i = (*arg_sensor_i ? arg_sensor_i->as_int() : 0);

        bool parallel_loading = !(llvm || cuda) || (*arg_load_par);

        // Append the mitsuba directory to the FileResolver search path list
        ref<Thread> thread   = Thread::thread();
        ref<FileResolver> fr = thread->file_resolver();
        fs::path base_path   = util::library_path().parent_path();
        if (!fr->contains(base_path))
            fr->append(base_path);

        // Append extra paths from command line arguments to the FileResolver
        // search path list
        if (*arg_paths) {
            auto extra_paths = string::tokenize(arg_paths->as_string(), ";");
            for (auto &path : extra_paths) {
                if (!fr->contains(path))
                    fr->append(path);
            }
        }

        if (!*arg_extra || *arg_help) {
            help((int) Thread::thread_count());
        } else {
            Log(Info, "%s", util::info_build((int) Thread::thread_count()));
            Log(Info, "%s", util::info_copyright());
            Log(Info, "%s", util::info_features());

#if !defined(NDEBUG)
            Log(Warn, "Renderer is compiled in debug mode, performance will be "
                      "considerably reduced.");
#endif
        }

        while (arg_extra && *arg_extra) {
            std::cout << "starting"<<std::endl;
            fs::path filename(arg_extra->as_string());
            ref<FileResolver> fr2 = new FileResolver(*fr);
            thread->set_file_resolver(fr2);

            // Add the scene file's directory to the search path.
            fs::path scene_dir = filename.parent_path();
            if (!fr2->contains(scene_dir))
                fr2->append(scene_dir);

            if (*arg_output)
                filename = arg_output->as_string();

            // parse first scene from the passed file.
            std::vector<ref<Object>> parsed_orig =
                xml::load_file(arg_extra->as_string(), mode, params,
                               *arg_update, parallel_loading);
            if (parsed_orig.size() != 1)
                Throw("Root element of the input file is expanded into "
                      "multiple objects, only a single object is expected!");

            MI_INVOKE_VARIANT(mode, render, parsed_orig[0].get(),
                                sensor_i,
                                filename);

            arg_extra = arg_extra->next();
            std::cout << "done with function" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cout << "error 1" << std::endl;
        error_msg = std::string("Caught a critical exception: ") + e.what();
    } catch (...) {
        std::cout << "error 2" << std::endl;
        error_msg = std::string("Caught a critical exception of unknown type!");
    }


    if (!error_msg.empty()) {
        /* Strip zero-width spaces from the message (Mitsuba uses these
           to properly format chains of multiple exceptions) */
        const std::string zerowidth_space = "\xe2\x80\x8b";
        while (true) {
            auto it = error_msg.find(zerowidth_space);
            if (it == std::string::npos)
                break;
            error_msg = error_msg.substr(0, it) + error_msg.substr(it + 3);
        }

#if defined(_WIN32)
        HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO console_info;
        GetConsoleScreenBufferInfo(console, &console_info);
        SetConsoleTextAttribute(console, FOREGROUND_RED | FOREGROUND_INTENSITY);
#else
        std::cerr << "\x1b[31m";
#endif
        std::cerr << std::endl << error_msg << std::endl;
#if defined(_WIN32)
        SetConsoleTextAttribute(console, console_info.wAttributes);
#else
        std::cerr << "\x1b[0m";
#endif
    }

    MI_INVOKE_VARIANT(mode, scene_static_accel_shutdown);
    color_management_static_shutdown();
    Profiler::static_shutdown();
    Bitmap::static_shutdown();
    Logger::static_shutdown();
    Thread::static_shutdown();
    Class::static_shutdown();
    Jit::static_shutdown();

#if defined(MI_ENABLE_CUDA)
    if (string::starts_with(mode, "cuda_")) {
        printf("%s\n", jit_var_whos());
        jit_shutdown();
    }
#endif

#if defined(MI_ENABLE_LLVM)
    if (string::starts_with(mode, "llvm_")) {
        printf("%s\n", jit_var_whos());
        jit_shutdown();
    }
#endif
    std::cout << "finished running" << std::endl;

    return error_msg.empty() ? 0 : -1;
}
