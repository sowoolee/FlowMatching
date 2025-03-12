import torch

from params_proto import ParamsProto

class Config(ParamsProto):
    prefix = "flow/4gait/unet"
    # misc
    seed = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bucket = '/home/hubolab/workspace/FM/weights/'
    dataset = 'go1-locomotion'

    ## model
    model = 'models.TemporalUnet'
    flow = 'models.CondOTFlowMatching'
    horizon = 56
    n_diffusion_steps = 100
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = False
    dim_mults = (1, 2, 4)
    returns_condition = True
    calc_energy=False
    dim = 128 # 128
    condition_dropout = 0.25
    condition_guidance_w = 1.4
    test_ret = 0.9
    renderer = 'utils.RaisimRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'LimitsNormalizer'
    clip_denoised = True
    use_padding = True
    include_returns = True
    max_path_length = 250
    hidden_dim = 512 # 256 # inv network dimension
    ar_inv = False
    train_only_inv = False
    action_scale = 1.0

    ## training
    n_train_steps = 100001
    n_steps_per_epoch = 100001
    loss_type = 'l2'
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 25000
    record_freq = 25000
    sample_freq = 1000
    eval_freq = 1000
    n_saves = 5
    save_parallel = False
    n_reference = 4
    save_checkpoints = True