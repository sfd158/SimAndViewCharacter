#pragma once
#include "Common.h"

class LinearLayer
{    
private:
    MatrixXf weight;
    VectorXf bias;

public:
    LinearLayer();
    LinearLayer(int input_size, int output_size);
    void ZeroInit(int input_size, int output_size);
    void XavierInit(int input_size, int output_size);
    void load_parameters(
        const std::string& weight_file,
        const std::string& bias_file
    );
    size_t load_weights(const float* data);

    void forward(const MatrixXf* input, MatrixXf* out) const;
};

// use ELU activate function for default.
class MLP
{
private:
    std::vector<LinearLayer> layers;
    int in_dim = 0, out_dim = 0, hid_dim = 0;
    MatrixXf* buf1 = nullptr; // save hidden results
    MatrixXf* buf2 = nullptr;

public:
    MLP();
    void Init(int in_dim_, int out_dim_, int hid_dim_, int nlayers);
    ~MLP();
    void forward(const MatrixXf* x, MatrixXf* out);
    size_t load_weights(const float* data);
};

class Encoder
{
private:
    std::vector<LinearLayer> fc_layers;
    LinearLayer mu_net;
    float var = 0.3f;
    MatrixXf buf1, buf2, buf3;

public:
    MatrixXf z, mu; // z is latent vector, and mu is mean of latent distribution

public:
    Encoder();
    Encoder(int input_size, int condition_size, int output_size,
        int hidden_layer_num, int hidden_layer_size, float var_);
    void Init(int input_size, int condition_size, int output_size,
        int hidden_layer_num, int hidden_layer_size, float var_);
    void encode(const MatrixXf* x, const MatrixXf* c);

    size_t load_weights(const float* data);
};

class MoELayer
{
private:
    std::vector<MatrixXf> weights; // num_experts, (in_dim, out_dim)
    MatrixXf bias; // (out_dim, num_experts)
    int in_dim = 0, out_dim = 0;
    bool use_layer_norm = true;
    bool use_elu = false;

public:
    MoELayer();
    MoELayer(int in_dim, int out_dim, int num_experts, 
        bool use_layer_norm_, bool use_elu_);
    MoELayer * Init(int in_dim, int out_dim, int num_experts,
        bool use_layer_norm_, bool use_elu_);
    void ZeroInit();
    void forward(const MatrixXf* coef, MatrixXf* xin, MatrixXf* result);
    size_t load_weights(const float* data);
};

void layer_norm_single(MatrixXf* res, float epsilon = 1e-5f);

void layer_norm(const MatrixXf* input, MatrixXf* res, float epsilon = 1e-5f);

class GatingMixedDecoder
{
private:
    MLP gate_net;
    MatrixXf xin, coef, action;
    MatrixXf buf1, buf2;
    std::vector<MoELayer> layers;
    int in_dim = 0, out_dim = 0;

public:
    GatingMixedDecoder();
    void Init(int latent_size, int condition_size, int output_size,
        int hidden_size, int num_experts, int num_layer, int gate_hsize);

    MatrixXf & forward(const MatrixXf* z, const MatrixXf* c);
    size_t load_weights(const float* data);
};

class PriorEncoder
{
public:
    Encoder prior, posterior;
    MatrixXf z, mu;
public:
    PriorEncoder();

    void Init(
        int input_size,
        int condition_size,
        int output_size,
        int hidden_layer_num,
        int hidden_layer_size,
        float fix_var);

    size_t load_weights(const float* data);
    MatrixXf& forward(const MatrixXf* n_obs, const MatrixXf* n_target);
    MatrixXf& act_prior(const MatrixXf* n_obs);
};

class ControlVAE
{
public:
    MatrixXf obs_mean, obs_std;
    // action is axis angle (num_joint * 3,)
    // action_quat is quaternion.
    MatrixXf n_obs, n_target, action_quat;
    PriorEncoder encoder;
    GatingMixedDecoder agent; // (latent, condition) => action
public:
    ControlVAE();
    ControlVAE * Init(int obs_size = 323,
        int latent_size = 64,
        int action_size = 57,
        int encoder_hidden_layer_num = 2,
        int encoder_hidden_layer_size = 512,
        int actor_num_experts = 6,
        int actor_hidden_layer_size = 512,
        int actor_hidden_layer_num = 3,
        int actor_gate_hidden_layer_size = 64,
        float var = 0.3f);

    size_t load_weights(const char* file_name);
    size_t load_weights(const float* data);
    void normalize(const MatrixXf* obs, MatrixXf * res);
    MatrixXf& act_prior(const MatrixXf* obs);
    MatrixXf& act_tracking(const MatrixXf * obs, const MatrixXf * target);

    MatrixXf& axis_angle_to_quat(const MatrixXf& act);
    MatrixXf& act_prior_quat(const MatrixXf* obs);
    MatrixXf& act_tracking_quat(const MatrixXf* obs, const MatrixXf* target);
};