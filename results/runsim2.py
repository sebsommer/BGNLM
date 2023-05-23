import sys
import os.path

if not os.path.isfile(f'feat_{sys.argv[1]}_{sys.argv[6]}.txt'):
    h = open(f'feat_{sys.argv[1]}_{sys.argv[6]}.txt', 'w')
    k = open(f'mprobs_{sys.argv[1]}_{sys.argv[6]}.txt', 'w')

    h.write(f'Features for dataset {sys.argv[1]} ({sys.argv[6]})\n\n')
    k.write(f'Mprobs for dataset {sys.argv[1]} ({sys.argv[6]})\n\n')

    h.close()
    k.close()

for i in range(int(sys.argv[5])):
    h = open(f'feat_{sys.argv[1]}_{sys.argv[6]}.txt', 'a')
    k = open(f'mprobs_{sys.argv[1]}_{sys.argv[6]}.txt', 'a')

    print(f"run {i+1}:")
    with open("main_new.py") as f:
        exec(f.read())
    h.write('{}\n'.format(best_gen['features'][best_gen['mprobs'] > 0.5]))
    k.write('{}\n'.format(best_gen['mprobs'][best_gen['mprobs'] > 0.5]))

    f.close()
    h.close()
    k.close()