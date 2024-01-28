import functools
import os
import multiprocessing
import learn


def doMultithread(process_id):
    stream_length = process_id + 1  # Adjust stream_length based on the process ID
    learn_rate = 0.0001
    model_num = process_id
    in_l, hid_ls, out_l = 6 * stream_length, [5, 5, 5], 1
    # learn(in_l, hid_ls, out_l, stream_length, learn_rate, model_num)

    # Use partial to create a new function with fixed parameters
    partial_learn = functools.partial(learn.learn, param1=in_l, param2=hid_ls, param3=out_l,
                                      param4=stream_length, param5=learn_rate, param6=model_num)

    # Call the partial_learn function
    partial_learn()


if __name__ == "__main__":
    num_threads = os.cpu_count()
    if num_threads < 21:
        raise ValueError("This program would crash your pc.")
    # Create a Pool with the specified number of processes
    with multiprocessing.Pool(processes=20) as pool:
        # Map the doMultithread function to each process ID
        pool.map(doMultithread, range(20))
