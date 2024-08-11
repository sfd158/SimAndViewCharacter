#include "Common.h"
#include "Network.h"
#include <fstream>

// elu activate function. alpha = 1 here.
static void elu(MatrixXf* ptr)
{
    (*ptr).noalias() = ((*ptr).array().cwiseMax(0.0f) + (*ptr).array().cwiseMin(0.0f).exp() - 1.0f).matrix();
}

// Linear layer
LinearLayer::LinearLayer() {}

LinearLayer::LinearLayer(int input_size, int output_size)
{
    XavierInit(input_size, output_size);
}

// initialize weight of network by Zeros
void LinearLayer::ZeroInit(int input_size, int output_size)
{
    weight = MatrixXf::Zero(input_size, output_size);
    bias = VectorXf::Zero(output_size);
}

// xavier initialize.
void LinearLayer::XavierInit(int input_size, int output_size)
{
    // Xavier initialization for weights
    float limit = std::sqrtf(6.0f / (input_size + output_size));
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);
    weight = MatrixXf::NullaryExpr(input_size, output_size, [&]() {return dis(gen);});
    bias = VectorXf::NullaryExpr(output_size, [&]() {return dis(gen);});
}

// load parameters from file
void load_matrixf_parameters(
    MatrixXf * weight,
    const std::string& weight_file
)
{
    std::ifstream weight_in(weight_file, std::ios::binary);
    if (!weight_in.is_open()) throw std::runtime_error("Unable to open file: " + weight_file);
    size_t r = weight->rows(), c = weight->cols();
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            weight_in.read(reinterpret_cast<char*>(&(*weight)(i, j)), sizeof(float));
    weight_in.close();
}

void load_vectorf_parameters(
    VectorXf* bias,
    const std::string& bias_file
)
{
    std::ifstream bias_in(bias_file, std::ios::binary);
    if (!bias_in.is_open()) throw std::runtime_error("Unable to open file: " + bias_file);
    size_t l = bias->size();
    for (size_t i = 0; i < l; ++i)
        bias_in.read(reinterpret_cast<char*>(&(*bias)(i)), sizeof(float));
    bias_in.close();
}

void LinearLayer::load_parameters(
    const std::string& weight_file, 
    const std::string& bias_file
)
{
    load_matrixf_parameters(&weight, weight_file);
    load_vectorf_parameters(&bias, bias_file);
}

// forward function of linear network.
void LinearLayer::forward(const MatrixXf* input, MatrixXf* out) const
{
    out->noalias() = (*input * weight).rowwise() + bias.transpose();
}

// load network weight from float pointer.
size_t LinearLayer::load_weights(const float* data)
{
    memcpy(weight.data(), data, sizeof(float) * weight.size());
    memcpy(bias.data(), data + weight.size(), sizeof(float) * bias.size());
    return weight.size() + bias.size();
}

MLP::MLP() {}

MLP::~MLP()
{
    // dealloc memory
    if (buf1 != nullptr) delete buf1;
    if (buf2 != nullptr) delete buf2;
}

void MLP::Init(int in_dim_, int out_dim_, int hid_dim_, int nlayers)
{
    in_dim = in_dim_;
    out_dim = out_dim_;
    hid_dim = hid_dim_;
    if (nlayers == 1) hid_dim = out_dim_;
    layers.resize(nlayers);
    layers[0].ZeroInit(in_dim, hid_dim);
    if (nlayers > 1)
    {
        for (int i = 1; i < nlayers - 1; i++) layers[i].ZeroInit(hid_dim, hid_dim);
        layers[nlayers - 1].ZeroInit(hid_dim, out_dim);
        buf1 = new MatrixXf(1, hid_dim);
        buf2 = new MatrixXf(1, hid_dim);
        // buf1 and buf2 are used to save hidden result 
    }
}

// forward function of MLP network
void MLP::forward(const MatrixXf* x, MatrixXf* out)
{
    size_t nl = layers.size();
    layers[0].forward(x, buf1);
    if (nl == 1) return;
    for (int i = 1; i < nl - 1; i++)
    {
        elu(buf1);
        layers[i].forward(buf1, buf2);
        std::swap(buf1, buf2);
    }
    elu(buf1);
    layers[nl - 1].forward(buf1, out);
}

// load network weights from float pointer.
size_t MLP::load_weights(const float* data)
{
    size_t off = 0;
    for (auto& layer : layers)
        off += layer.load_weights(data + off);
    return off;
}

// layer norm. for simple, assume input shape is (1, channels)
void layer_norm_single(MatrixXf* res, float epsilon)
{
    eigen_assert(res->rows() == 1);
    float mean = res->mean();
    float var = (res->array() - mean).square().mean();
    res->array() = (res->array() - mean) / std::sqrt(var + epsilon);
}

// layer norm. The input shape is (batch, channel)
void layer_norm(const MatrixXf* input, MatrixXf* res, float epsilon)
{
    Eigen::Index cols = input->cols();
    VectorXf mean = input->rowwise().mean();  // mean for each row

    // Compute variance for each row
    MatrixXf centered = input->colwise() - mean;
    VectorXf variance = (centered.array().square().rowwise().sum() / cols).matrix();

    res->noalias() = (centered.array().colwise() / (variance.array() + epsilon).sqrt()).matrix();
}

// mixture of expert layer in ControlVAE.
MoELayer::MoELayer() {}

MoELayer::MoELayer(int in_dim, int out_dim, int num_experts,
    bool use_layer_norm_, bool use_elu_)
{
    Init(in_dim, out_dim, num_experts, use_layer_norm_, use_elu_);
}

// Initialize network weight by zero
MoELayer * MoELayer::Init(int in_dim, int out_dim, int num_experts, bool use_layer_norm_, bool use_elu_)
{
    use_layer_norm = use_layer_norm_;
    use_elu = use_elu_;
    weights.resize(num_experts);
    for (int i = 0; i < num_experts; i++)
    {
        weights[i] = MatrixXf(in_dim, out_dim);
    }
    bias = MatrixXf(num_experts, out_dim);
    ZeroInit();
    return this;
}

void MoELayer::ZeroInit()
{
    for (int i = 0; i < weights.size(); i++) weights[i].setZero();
    bias.setZero();
}

// forward operation of MoE
void MoELayer::forward(const MatrixXf* coef, MatrixXf* xin, MatrixXf* result)
{
    if (this->use_layer_norm) layer_norm_single(xin);
    size_t n_experts = weights.size();
    // contract('e,ek->k', coefficients, bias)
    // coef == (1, n_expert), bias = (n_expert, out_dim),  xin == (1, in_dim)
    result->noalias() = (*coef) * bias;
    // contract('e, j, ejk->k', coefficients, input, weight)
    for (int e = 0; e < n_experts; ++e)
    {
        result->noalias() += (*coef)(e) * ((*xin) * weights[e]);
    }
    if (use_elu) elu(result);
}

// load network weight from float pointer.
size_t MoELayer::load_weights(const float* data)
{
    size_t off = 0;
    for (auto& w : weights)
    {
        memcpy(w.data(), data + off, sizeof(float) * w.size());
        off += w.size();
    }
    memcpy(bias.data(), data + off, sizeof(float) * bias.size());
    return off + bias.size();
}

// MoE decoder in ControlVAE.
GatingMixedDecoder::GatingMixedDecoder() {}

// initialize MoE decoder.
void GatingMixedDecoder::Init(
    int latent_size, int condition_size, int output_size,
    int hidden_size, int num_experts, int num_layer, int gate_hsize)
{
    int inter_size = latent_size + hidden_size;
    in_dim = latent_size + condition_size;
    out_dim = output_size;

    coef = MatrixXf::Zero(1, num_experts);
    if (num_layer > 1)
    {
        buf1 = MatrixXf::Zero(1, hidden_size); // for compute hidden results
        buf2 = MatrixXf::Zero(1, inter_size);
    }
    
    xin = MatrixXf::Zero(1, in_dim);
    gate_net.Init(in_dim, num_experts, gate_hsize, 3);
    layers.resize(num_layer + 1);
    if (num_layer > 1)
    {
        layers[0].Init(in_dim, hidden_size, num_experts, true, true);
        for (int i = 1; i < num_layer; i++)
        {
            layers[i].Init(inter_size, hidden_size, num_experts, true, true);
        }
    }
    layers[num_layer].Init(inter_size, output_size, num_experts, true, false);
}

// concatenate matrix a and b.
void ColCat(const MatrixXf* a, const MatrixXf* b, MatrixXf * res)
{
    res->leftCols(a->cols()) = *a;
    res->rightCols(b->cols()) = *b;
}

MatrixXf & GatingMixedDecoder::forward(
    const MatrixXf* z, const MatrixXf* c)
{
    ColCat(z, c, &xin);

    // compute coef of gating network
    gate_net.forward(&xin, &coef);
    auto expo = (coef.array() - coef.maxCoeff()).exp();
    coef.array() = expo / expo.sum(); // softmax

    if (layers.size() == 1)
    {
        layers[0].forward(&coef, &xin, &action);
        return action;
    }
    layers[0].forward(&coef, &xin, &buf1);

    for (int i = 1; i < layers.size(); i++)
    {
        ColCat(z, &buf1, &buf2);
        layers[i].forward(&coef, &buf2, &buf1);
    }
    action.noalias() = buf1;
    return action;
}

// load network weight from float pointer.
size_t GatingMixedDecoder::load_weights(const float* data)
{
    size_t off = gate_net.load_weights(data);
    for (auto& layer : layers)
    {
        off += layer.load_weights(data + off);
    }
    return off;
}

// encoder network of controlvae.
Encoder::Encoder() {}

Encoder::Encoder(int input_size, int condition_size, int output_size,
    int hidden_layer_num, int hidden_layer_size, float var_)
{
    Init(input_size, condition_size, output_size,
        hidden_layer_num, hidden_layer_size, var_);
}

// initialize network weight by zero.
void Encoder::Init(int input_size, int condition_size, int output_size,
    int hidden_layer_num, int hidden_layer_size, float var_)
{
    var = var_; // the variance of vae is fixed in ControlVAE
    fc_layers.resize(hidden_layer_num + 1);
    fc_layers[0].ZeroInit(input_size + condition_size, hidden_layer_size);
    for (int i = 1; i < hidden_layer_num + 1; i++)
    {
        fc_layers[i].ZeroInit(input_size + hidden_layer_size, hidden_layer_size);
    }
    mu_net.ZeroInit(input_size + hidden_layer_size, output_size);
    // buf1, buf2, buf3 are used for saving hidden results
    buf1 = MatrixXf::Zero(1, hidden_layer_size);
    buf2 = MatrixXf::Zero(1, input_size + hidden_layer_size);
    buf3 = MatrixXf::Zero(1, input_size + condition_size);
    z = MatrixXf::Zero(1, output_size); // latent vector
    mu = MatrixXf::Zero(1, output_size); // mean value of latent distribution.
}

void Encoder::encode(const MatrixXf* x, const MatrixXf* c)
{
    // for prior network, the condition (c) is null
    // for posterior network, the input is x and c.
    if (c == nullptr) fc_layers[0].forward(x, &buf1); // prior
    else // posterior
    {
        ColCat(x, c, &buf3); // concatenate input x and c.
        fc_layers[0].forward(&buf3, &buf1);
    }
    elu(&buf1);
    for (size_t i = 1; i < fc_layers.size(); i++)
    {
        ColCat(x, &buf1, &buf2);
        fc_layers[i].forward(&buf2, &buf1);
        elu(&buf1);
    }
    ColCat(x, &buf1, &buf2);
    mu_net.forward(&buf2, &mu);

    // generate random noise.
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> d(0.0f, 1.0f);
    // if (c != nullptr)
    {
        for (Eigen::Index j = 0; j < z.cols(); ++j)
            z(0, j) = d(gen) + mu(0, j);
    }
}

// load network weight from data pointer.
size_t Encoder::load_weights(const float* data)
{
    size_t off = 0;
    for (auto& layer : fc_layers)
        off += layer.load_weights(data + off);
    off += mu_net.load_weights(data + off);
    return off;
}

PriorEncoder::PriorEncoder() {}

void PriorEncoder::Init(
    int input_size,
    int condition_size,
    int output_size,
    int hidden_layer_num,
    int hidden_layer_size,
    float fix_var)
{
    prior.Init(condition_size, 0, output_size, hidden_layer_num, hidden_layer_size, fix_var);
    posterior.Init(input_size, condition_size, output_size, hidden_layer_num, hidden_layer_size, fix_var);
    z = MatrixXf::Zero(1, output_size);
    mu = MatrixXf::Zero(1, output_size);
}

size_t PriorEncoder::load_weights(const float* data)
{
    size_t off = prior.load_weights(data);
    off += posterior.load_weights(data + off);
    return off;
}

// input:
// n_obs is normalized observation of current time.
// n_target is normalized desired target pose of next time.
MatrixXf& PriorEncoder::forward(const MatrixXf * n_obs, const MatrixXf * n_target)
{
    prior.encode(n_obs, nullptr);
    posterior.encode(n_target, n_obs);
    z.noalias() = prior.mu + posterior.z; // latent vector
    mu.noalias() = prior.mu + posterior.mu; // mean of latent distribution.
    return z;
}

// n_obs is normalized observation of current time.
MatrixXf& PriorEncoder::act_prior(const MatrixXf* n_obs)
{
    prior.encode(n_obs, nullptr);
    return prior.z;
}

ControlVAE::ControlVAE() {}

// Initialize ControlVAE network.
ControlVAE * ControlVAE::Init(int obs_size, int latent_size, int action_size, 
    int encoder_hidden_layer_num,
    int encoder_hidden_layer_size,
    int actor_num_experts,
    int actor_hidden_layer_size,
    int actor_hidden_layer_num,
    int actor_gate_hidden_layer_size,
    float var)
{
    encoder.Init(obs_size, obs_size, latent_size,
        encoder_hidden_layer_num, encoder_hidden_layer_size, var);
        
    agent.Init(latent_size, obs_size, action_size, actor_hidden_layer_size,
        actor_num_experts, actor_hidden_layer_num,actor_gate_hidden_layer_size);
    obs_mean = MatrixXf::Zero(1, obs_size);
    obs_std = MatrixXf::Ones(1, obs_size);
    // action = MatrixXf::Zero(1, action_size);
    action_quat = MatrixXf::Zero(action_size / 3, 4);
    return this;
}

// load network weight from file, which is exported from pytorch code.
size_t ControlVAE::load_weights(const char* file_name)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file) {
        std::cerr << "can not open " << file_name << std::endl;
        return 0;
    }
    file.seekg(0, file.end);
    size_t length = file.tellg();
    file.seekg(0, file.beg);
    std::vector<float> buffer(length / sizeof(float));
    file.read(reinterpret_cast<char*>(buffer.data()), length);
    file.close();
    size_t ret = load_weights(buffer.data());
    assert(ret == buffer.size());
    return ret;
}

size_t ControlVAE::load_weights(const float* data)
{
    size_t off = encoder.load_weights(data);
    off += agent.load_weights(data + off);
    obs_mean.noalias() = Eigen::Map<const MatrixXf>(data + off, obs_mean.rows(), obs_mean.cols());
    off += obs_mean.size();
    obs_std.noalias() = Eigen::Map<const MatrixXf>(data + off, obs_std.rows(), obs_std.cols());
    off += obs_std.size();
    return off;
}

// normalize observation.
void ControlVAE::normalize(const MatrixXf* obs, MatrixXf * res)
{
    res->noalias() = ((*obs - obs_mean).array() / (obs_std.array() + 1e-8)).matrix();
}

// input: observation of current time.
// output: action in axis angle format.
MatrixXf& ControlVAE::act_prior(const MatrixXf * obs)
{
    normalize(obs, &n_obs);
    auto & z = encoder.act_prior(&n_obs);
    return agent.forward(&z, &n_obs);
}

// input:
// obs: observation of current time.
// target: desired target pose of next time.
MatrixXf& ControlVAE::act_tracking(const MatrixXf* obs, const MatrixXf* target)
{
    normalize(target, &n_target);
    normalize(obs, &n_obs);
    auto & z = encoder.forward(&n_obs, &n_target); // z is latent vector
    return agent.forward(&z, &n_obs); // return action in axis angle format
}

MatrixXf& ControlVAE::axis_angle_to_quat(const MatrixXf& act)
{
    // TODO: check.
    for (Eigen::Index i = 0; i < action_quat.rows(); i++)
    {
        auto block = act.block(0, i * 3, 1, 3);
        auto norm = block.norm();
        auto q = Eigen::Quaternionf(Eigen::AngleAxisf(norm, block / norm));
        action_quat.block(i, 0, 1, 4) = q.coeffs();
    }
    return action_quat;
}

MatrixXf& ControlVAE::act_prior_quat(const MatrixXf* obs)
{
    return axis_angle_to_quat(act_prior(obs));
}

MatrixXf& ControlVAE::act_tracking_quat(const MatrixXf* obs, const MatrixXf* target)
{
    return axis_angle_to_quat(act_tracking(obs, target));
}

#if 1
extern "C" // wrapper of network.
{
    EXPORT_API LinearLayer* LinearLayer_new(int input_size, int output_size)
    {
        return new LinearLayer(input_size, output_size);
    }

    EXPORT_API void LinearLayer_del(LinearLayer* obj)
    {
        delete obj;
    }

    EXPORT_API void LinearLayer_forward(LinearLayer* obj, const MatrixXf* x, MatrixXf * out)
    {
        obj->forward(x, out);
    }

    EXPORT_API MLP* MLP_new(int in_dim_, int out_dim_, int hid_dim_, int nlayers)
    {
        MLP* ret = new MLP();
        ret->Init(in_dim_, out_dim_, hid_dim_, nlayers);
        return ret;
    }

    EXPORT_API void MLP_del(MLP* mlp)
    {
        if (mlp != nullptr) delete mlp;
    }

    EXPORT_API void MLP_forward(MLP * mlp, const MatrixXf* x, MatrixXf* out)
    {
        mlp->forward(x, out);
    }

    EXPORT_API MoELayer * MoELayer_new(
        int in_dim, int out_dim, int num_experts, bool use_layer_norm, bool use_elu)
    {
        return (new MoELayer())->Init(in_dim, out_dim, num_experts, use_layer_norm, use_elu);
    }

    EXPORT_API void MoELayer_del(MoELayer* ptr)
    {
        if (ptr != nullptr) delete ptr;
    }

    EXPORT_API void MoELayer_forward(MoELayer * ptr, MatrixXf* coef, MatrixXf* xin, MatrixXf* result)
    {
        ptr->forward(coef, xin, result);
    }

    EXPORT_API ControlVAE* ControlVAE_new()
    {
        return (new ControlVAE())->Init();
    }

    EXPORT_API void ControlVAE_del(ControlVAE* ptr)
    {
        if (ptr != nullptr) delete ptr;
    }

    EXPORT_API size_t ControlVAE_load_weights(ControlVAE * ptr, const char * fname)
    {
        return ptr->load_weights(fname);
    }

    EXPORT_API void ControlVAE_act_prior(ControlVAE* ptr, const MatrixXf* obs, MatrixXf * res)
    {
        res->noalias() = ptr->act_prior(obs);
    }

    EXPORT_API void ControlVAE_act_prior_quat(ControlVAE* ptr, const MatrixXf* obs, MatrixXf* res)
    {
        res->noalias() = ptr->act_prior_quat(obs);
    }

    EXPORT_API void ControlVAE_act_tracking(ControlVAE * ptr, const MatrixXf* obs, const MatrixXf* target, MatrixXf* res)
    {
        res->noalias() = ptr->act_tracking(obs, target);
    }

    EXPORT_API void ControlVAE_act_tracking_quat(ControlVAE* ptr, const MatrixXf* obs, const MatrixXf* target, MatrixXf* res)
    {
        res->noalias() = ptr->act_tracking_quat(obs, target);
    }
}
#endif
