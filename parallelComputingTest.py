import multiprocessing

def func(index: int):
    print(index)

if __name__ == '__main__':
    processes = []

    for i in range(0, 6):
        p = multiprocessing.Process(target=func, args=(i,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()