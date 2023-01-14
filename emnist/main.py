import sys
import getopt

from train import train


def main(argv):
    epochs = 10
    batch_size = 10000
    learning_rate = 0.6
    shuffle = False
    input_count = 784
    hidden_count = 80
    output_count = 10
    report_path = "report.csv"
    config_path = "config.txt"
    with_gpu = False
    opts, args = getopt.getopt(argv, "h", [
                               "epochs=", "batch_size=", "learning_rate=", "shuffle=", "input_count=", "hidden_count=", "output_count=", "report_path=", "config_path=", "with_gpu="])
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -e <epochs> -bs <batch_size> -lr <learning_rate> -s <shuffle> -ic <input_count> -hc <hidden_count> -oc <output_count> -cp <config_path> -rp <report_path> -wg <with_gpu>')
            sys.exit()
        elif opt in ("--epochs"):
            epochs = int(arg)
        elif opt in ("--batch_size"):
            batch_size = int(arg)
        elif opt in ("--learning_rate"):
            learning_rate = float(arg)
        elif opt in ("--shuffle"):
            shuffle = bool(arg)
        elif opt in ("--input_count"):
            input_count = int(arg)
        elif opt in ("--hidden_count"):
            hidden_count = int(arg)
        elif opt in ("--output_count"):
            output_count = int(arg)
        elif opt in ("--report_path"):
            report_path = str(arg)
        elif opt in ("--config_path"):
            config_path = str(arg)
        elif opt in ("--with_gpu"):
            with_gpu = bool(arg)
    print('epochs: ', epochs)
    print('batch_size: ', batch_size)
    print('learning_rate: ', learning_rate)
    print('shuffle: ', shuffle)
    print('input_count: ', input_count)
    print('hidden_count: ', hidden_count)
    print('output_count: ', output_count)
    print('report_path: ', report_path)
    print('config_path: ', config_path)
    print('with_gpu: ', with_gpu)

    train(epochs, batch_size, learning_rate, shuffle,
          input_count, hidden_count, output_count, report_path, config_path)


if __name__ == "__main__":
    main(sys.argv[1:])
