import libraries.Intelligence as intLib
import app.dataParse as dp
import os


learn_rate = 0.001
file_path = os.path.join(os.path.abspath(".."), "data", "stocks")


def init(in_l, hid_ls, out_l):
    return intLib.Intelligence(in_l, hid_ls, out_l, learn_rate)


def learn(in_l, hid_ls, out_l, stream_length):
    intel = init(in_l, hid_ls, out_l)

    file_index = 0

    file_cnt = dp.getCsvFiles(file_path)

    while (dataset := dp.load(file_path, file_index)) is not None:
        data_index = 0
        dataset = intLib.flatten(dataset)
        while (data := dp.parse(dataset, data_index, stream_length)) is not None:
            data_index += 1
            if float(dataset[(stream_length+data_index-1)*6]) > 0 and float(dataset[(stream_length+data_index)*6]) > 0:

                target = float(dataset[(stream_length+data_index)*6]) / float(dataset[(stream_length+data_index-1)*6])

                if abs(target) < 5:
                    intel.train(data, target)
        file_index += 1
        print(f"progress: {file_index} out of: {file_cnt}. {100.0*file_index/file_cnt}% complete")
        if file_index > 0.8 * file_cnt:
            break
    intLib.export_intelligence(intel, "output.csv")


if __name__ == "__main__":

    stream_length = 1
    in_l, hid_ls, out_l = 6 * stream_length, [6, 6, 6], 1
    learn(in_l, hid_ls, out_l, stream_length)
