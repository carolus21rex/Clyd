import os
import threading
import learn


def doMultithread():
    stream_length = 1
    learn_rate = 0.0001
    model_num = 0
    in_l, hid_ls, out_l = 6 * stream_length, [5, 5, 5], 1
    # learn(in_l, hid_ls, out_l, stream_length, 0.001, 0)

    threads = []
    for _ in range(20):
        model_num += 1
        thread = threading.Thread(target=learn.learn, args=(in_l, hid_ls, out_l, stream_length, learn_rate, model_num))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    num_threads = os.cpu_count()
    if num_threads < 21:
        raise ValueError("This program would crash your pc.")
    doMultithread()
