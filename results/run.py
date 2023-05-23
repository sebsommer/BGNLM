import sys
import os.path

if not os.path.isfile(f'results{sys.argv[1]}{sys.argv[4]}_MF.txt'):
    g = open(f'results{sys.argv[1]}{sys.argv[4]}_MF.txt', "w")
    h = open(f'feat{sys.argv[1]}{sys.argv[4]}_MF.txt', 'w')
    k = open(f'mprobs{sys.argv[1]}{sys.argv[4]}_MF.txt', 'w')
    l = open(f'all_feat{sys.argv[1]}{sys.argv[4]}_MF.txt', 'w')
    m = open(f'all_mprobs{sys.argv[1]}{sys.argv[4]}_MF.txt', 'w')

    g.write(f'Results for dataset {sys.argv[1]} ({sys.argv[4]}) (Mean-field):\n\n')
    g.write("{:<35} {:<35} {:<35}\n".format("RMSE:", "MAE:", "Corr:"))
    h.write(f'Best features (mprob >0.3) for dataset {sys.argv[1]} ({sys.argv[4]}) (Mean-field):\n\n')
    k.write(f'Best mprobs (mprob > 0.3) for dataset {sys.argv[1]} ({sys.argv[4]}) (Mean-field)\n\n')
    l.write(f'All features for dataset {sys.argv[1]} ({sys.argv[4]}) (Mean-field):\n\n')
    m.write(f'All mprobs for dataset {sys.argv[1]} ({sys.argv[4]}) (Mean-field)\n\n')


    g.close()
    h.close()
    k.close()

for i in range(int(sys.argv[5])):
    g = open(f'results{sys.argv[1]}{sys.argv[4]}_MF.txt', "a")
    h = open(f'feat{sys.argv[1]}{sys.argv[4]}_MF.txt', 'a')
    k = open(f'mprobs{sys.argv[1]}{sys.argv[4]}_MF.txt', 'a')
    l = open(f'all_feat{sys.argv[1]}{sys.argv[4]}_MF.txt', 'a')
    m = open(f'all_mprobs{sys.argv[1]}{sys.argv[4]}_MF.txt', 'a')


    print(f"run {i+1}:")
    with open("main_mf.py") as f:
        exec(f.read())
    a, b, c = best_gen['RMSE'], best_gen['MAE'], best_gen['corr']
    g.write("{:<35} {:<35} {:<35}\n".format(a, b, c))
    h.write('{}\n'.format(best_gen['features'][best_gen['mprobs'] > 0.3]))
    k.write('{}\n'.format(best_gen['mprobs'][best_gen['mprobs'] > 0.3]))
    l.write('{}\n'.format(best_gen['features']))
    m.write('{}\n'.format(best_gen['mprobs']))

    f.close()
    g.close()
    h.close()
    k.close()
    l.close()
    m.close()
