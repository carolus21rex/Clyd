import libraries.Intelligence as intLib
import app.dataParse as dp
import os



file_path = os.path.join(os.path.abspath(".."), "data", "stocks")


def init(in_l, hid_ls, out_l, learn_rate):
    return intLib.Intelligence(in_l, hid_ls, out_l, learn_rate)


def learn(in_l, hid_ls, out_l, stream_length, learn_rate, model_num):
    intel = init(in_l, hid_ls, out_l, learn_rate)

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
        # print(f"progress: {file_index} out of: {file_cnt}. {100.0*file_index/file_cnt}% complete")
        if file_index > 0.8 * file_cnt:
            break

    cnt = 0
    success = 0
    while (dataset := dp.load(file_path, file_index)) is not None:
        data_index = 0
        dataset = intLib.flatten(dataset)
        while (data := dp.parse(dataset, data_index, stream_length)) is not None:
            data_index += 1
            if float(dataset[(stream_length+data_index-1)*6]) > 0 and float(dataset[(stream_length+data_index)*6]) > 0:

                target = float(dataset[(stream_length+data_index)*6]) / float(dataset[(stream_length+data_index-1)*6])

                if abs(target) < 5:
                    out = intel.predict(data)
                    cnt += 1
                    if (out > 1 and target > 1) or (out < 1 and target < 1):
                        success += 1

        file_index += 1
        print(f"model: {model_num} progress: {file_index} out of: {file_cnt}. {100.0*file_index/file_cnt}% complete")
    intLib.export_intelligence(intel, f"model_{str(model_num).zfill(5)}_{format(success/cnt, '.5f').replace('.', '')}.csv")
