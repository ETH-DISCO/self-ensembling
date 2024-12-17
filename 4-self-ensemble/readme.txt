{
    #
    # config
    #

    "dataset": "cifar10",
    "training_noise": false,
    "training_shuffle": false,
    "training_adversarial": false,

    #
    # stuff that worked (keep from previous experiment)
    #

    "plain_layer_accs": {
        "20": 0.8211,
        "30": 0.8793,
        "35": 0.9103,
        "40": 0.9327,
        "45": 0.9437,
        "50": 0.9438,
        "52": 0.9498
    },
    "plain_ensemble_acc": 0.9458,
    "fgsm_20_ensemble_acc": 0.4165,

    "fgsm_20_layer_accs": {
        "20": 0.0129,
        "30": 0.2135,
        "35": 0.3429,
        "40": 0.4223,
        "45": 0.4833,
        "50": 0.5492,
        "52": 0.5511
    },
    ...
    "fgsm_52_ensemble_acc": 0.5841,
    "fgsm_52_layer_accs": {
        "20": 0.5688,
        "30": 0.5993,
        "35": 0.583,
        "40": 0.5735,
        "45": 0.5662,
        "50": 0.5437,
        "52": 0.5299
    },

    "fgsmcombined_[20, 30, 35]_ensemble_acc": 0.2928,
    "fgsmcombined_[20, 30, 35]_layer_accs": {
        "20": 0.0567,
        "30": 0.1278,
        "35": 0.2234,
        "40": 0.2965,
        "45": 0.3495,
        "50": 0.4267,
        "52": 0.4273
    },

    "fgsmensemble_ensemble_acc": 0.6741,
    "fgsmensemble_layer_accs": {
        "20": 0.5625,
        "30": 0.5302,
        "35": 0.5788,
        "40": 0.6167,
        "45": 0.6418,
        "50": 0.671,
        "52": 0.6761
    },

    #
    # stuff that didn't work
    #

    "mask_0_layer_accs": {
        "20": 0.1,
        "30": 0.1,
        "35": 0.1,
        "40": 0.1,
        "45": 0.1,
        "50": 0.1,
        "52": 0.1
    },
    "mask_0_ensemble_acc": 0.1,
    ...
    "mask_255_layer_accs": {
        "20": 0.1,
        "30": 0.1,
        "35": 0.1,
        "40": 0.1,
        "45": 0.1,
        "50": 0.1,
        "52": 0.1
    },
    "mask_255_ensemble_acc": 0.1
}