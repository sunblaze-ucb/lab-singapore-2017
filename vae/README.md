First download the pre-trained VAE-GAN model from [here](https://goo.gl/89s4BI) and
unpack it under `models/`.

To generate some attacks on the CelebA faces dataset using one of the attack methods:

```
python main.py faces vae_gan simple optimization_lvae \
               --only-existing --model-latent-dim 2048 \
               --optimization-lambda 0.75 --optimization-epochs 1000 --optimization-lr 0.1 \
               --attack-mode index --attack-no-reconstruct-target --attack-index 0
```

