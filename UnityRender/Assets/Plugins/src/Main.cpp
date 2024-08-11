#include <iostream>
#include <fstream>
#include "Network.h"

MatrixXf * load_from_file(const char* file_name)
{
	std::ifstream file(file_name, std::ios::binary);
	file.seekg(0, file.end);
	size_t length = file.tellg();
	file.seekg(0, file.beg);
	MatrixXf* res = new MatrixXf(1, length / sizeof(float));
	file.read(reinterpret_cast<char*>(res->data()), length);
	file.close();
	return res;
}

// check controlvae eigen version.
int main()
{
	ControlVAE vae;
	vae.Init();
	vae.load_weights("D:/test/Control-VAE/controlvae.bin");
	MatrixXf* nobs = load_from_file("D:/test/Control-VAE/n_obs.bin");
	MatrixXf* ntarget = load_from_file("D:/test/Control-VAE/n_target.bin");
	MatrixXf* mu_post = load_from_file("D:/test/Control-VAE/mu_post.bin");
	// vae.encoder.forward(nobs, ntarget);
	MatrixXf* z = load_from_file("D:/test/Control-VAE/latent_code.bin");
	MatrixXf* cal_act = load_from_file("D:/test/Control-VAE/action.bin");
	auto & act = vae.agent.forward(z, nobs);
	std::cout << act - *cal_act << std::endl;
	return 0;
}