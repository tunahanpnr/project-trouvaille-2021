import time

from Trouvaille import Trouvaille


def main():
    trouvaille = Trouvaille()
    trouvaille.run()


if __name__ == '__main__':
    start_time = time.time()

    main()

    print("run time: %s" % (time.time() - start_time))