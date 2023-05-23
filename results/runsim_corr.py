import sys
import os.path


if not os.path.isfile(f'feat_{sys.argv[1]}_{sys.argv[6]}_{sys.argv[7]}.txt'):
    g = open(f'all_mprobs{sys.argv[1]}_{sys.argv[6]}_{sys.argv[7]}.txt', "w")
    h = open(f'feat_{sys.argv[1]}_{sys.argv[6]}_{sys.argv[7]}.txt', 'w')
    k = open(f'mprobs_{sys.argv[1]}_{sys.argv[6]}_{sys.argv[7]}.txt', 'w')

    g.write(f'All mprobs for dataset {sys.argv[1]} ({sys.argv[6]}) (IAF layers = {sys.argv[7]}):\n\n')
    h.write(f'Features for dataset {sys.argv[1]} ({sys.argv[6]}) (IAF layers = {sys.argv[7]}):\n\n')
    k.write(f'Mprobs for dataset {sys.argv[1]} ({sys.argv[6]}) (IAF layers = {sys.argv[7]}):\n\n')

    g.close()
    h.close()
    k.close()

for i in range(int(sys.argv[5])):
    g = open(f'all_mprobs{sys.argv[1]}_{sys.argv[6]}_{sys.argv[7]}.txt', "a")
    h = open(f'feat_{sys.argv[1]}_{sys.argv[6]}_{sys.argv[7]}.txt', 'a')
    k = open(f'mprobs_{sys.argv[1]}_{sys.argv[6]}_{sys.argv[7]}.txt', 'a')

    print(f"run {i+1}:")
    with open("main_corr_iaf.py") as f:
        exec(f.read())
    g.write("{}\n".format(best_gen['mprobs']))
    h.write('{}\n'.format(best_gen['features'][best_gen['mprobs'] > 0.5]))
    k.write('{}\n'.format(best_gen['mprobs'][best_gen['mprobs'] > 0.5]))

    f.close()
    g.close()
    h.close()
    k.close()