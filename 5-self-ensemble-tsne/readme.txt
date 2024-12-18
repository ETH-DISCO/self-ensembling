tsne plots:
- instead of linear probes just dump the latents of models
- stack latents for all images (both perturbed and unperturbed)
- for a limited set of layers (maybe just the ones you do linear probes for, maybe just 9)
- make scatterplot for ~100 images
- add color/shape to encode whether they're perturbed/unperturbed and the class (you're using imagenette, not imagenet)

set all train configs to true / false simultaneously:
- "training_noise": [False, True],
- "training_shuffle": [False, True],
- "training_adversarial": [False, True],
