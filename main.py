import FileOpener
from Network import NeuralNetwork
import data_management
from logs import logger, debugger
# import save_network


def build_config(config_file, preset):
    def normalize_key():
        key_norm = key.lower().replace(' ', '_')
        match key_norm:
            case 'inp':
                return 'input'
            case 'out':
                return 'output'
            case 'hidden_layers' | 'hidden_layer':
                return 'h_layers'
            case 'logger':
                return 'log'
            case 'debugger':
                return 'debug'
            case 'activator':
                return 'activation'
            case _:
                return key_norm

    config = {}
    for key in config_file:
        if key == 'presets' or key == 'preset':
            continue
        config[normalize_key()] = config_file[key]

    for key in preset:
        config[normalize_key()] = preset[key]
    return config


def setup_global_logger(main_config: dict):
    if main_config.get('log', False):
        return logger.Log(main_config.get('file location', '') + '/' + main_config.get('file name', 'global_logger.txt'))
    return None


def write_strings(config, iteration):
    for key in config:
        if isinstance(config[key], str):
            config[key] = config[key].replace('{n}', f"{{{iteration}}}").replace('{\\n}', "{n}")
        elif isinstance(config[key], dict):
            config[key] = write_strings(config[key], iteration)

    return config


def setup_config(config, presets, i):
    new_config = write_strings(build_config(config, presets[i] if i < len(presets) else {}), i)
    setup_logger(new_config)
    return new_config


def setup_logger(config):
    config['log'] = logger.Log(config.get('log_file', 'logs/log.txt')) if config.get('log', False) else False


def get_accuracy(predictions, actual):
    right = 0
    wrong = 0
    for val1, val2 in zip(predictions, actual):
        if val1 != val2:
            wrong += 1
        else:
            right += 1
    return right, wrong, 100 * right / (right + wrong)


def running_neural_training(neural_network, config):
    neural_network.set_network(config=config)
    X, y = data_management.get_data(config['input_file'], config['output_file'], config['map_file'], randomize=config.get('randomize_data', False))
    training_groups_inp, training_groups_out, rest_inp, rest_out = data_management.split_into_groups(config.get('training_sets', 1), config.get('training_set_size', len(X)), X, y)

    for num, (inp, out) in enumerate(zip(training_groups_inp, training_groups_out)):
        neural_network.set_network(input=inp, output=out)
        neural_network.train(config.get("runs_per_set", 1))

    return get_accuracy(neural_network.think(rest_inp), rest_out)


def main():

    import time

    config = FileOpener.getConfig()
    main_config = config.pop('main') if 'main' in config else {}
    presets = config.pop('presets') if 'presets' in config else []

    global_logger = setup_global_logger(main_config)
    global_debug = debugger.Debug()

    start = time.time()

    n = NeuralNetwork.NeuralNetwork()
    for i in range(main_config.get('neural_networks_run', 1)):
        new_config = setup_config(config, presets, i)

        new_config['debug'] = global_debug.switch(new_config.get('debug', False))

        right, wrong, percent = running_neural_training(n, new_config)

        if new_config['log']:
            new_config['log'].close()

        if global_logger:
            global_logger.write(f"Right >> {right}, Wrong >> {wrong}\nNeural Accuracy >> {round(percent, main_config['accuracy_rounding'])}")
            global_logger.add('accuracy', [right, wrong, percent])

        global_debug.print(f"Right >> {right}, Wrong >> {wrong}\nNeural Accuracy >> {round(percent, main_config['accuracy_rounding'])}")

        pass

    print(time.time() - start)

    if global_logger:
        global_logger.close()
    pass


if __name__ == '__main__':

    main()
    pass
