import sys
import os.path

if not os.path.isfile(f'feat_{sys.argv[1]}_{sys.argv[6]}.txt'):
    #g = open(f'results_{sys.argv[1]}_{sys.argv[6]}.txt', "w")
    h = open(f'feat_{sys.argv[1]}_{sys.argv[6]}.txt', 'w')
    k = open(f'mprobs_{sys.argv[1]}_{sys.argv[6]}.txt', 'w')

    #g.write(f'Results for dataset {sys.argv[1]} ({sys.argv[6]}):\n\n')
    #g.write("{:<35} {:<35} {:<35}\n".format("RMSE:", "MAE:", "Corr:"))
    h.write(f'Features for dataset {sys.argv[1]} ({sys.argv[6]})\n\n')
    k.write(f'Mprobs for dataset {sys.argv[1]} ({sys.argv[6]})\n\n')

    #g.close()
    h.close()
    k.close()

for i in range(int(sys.argv[5])):
    #g = open(f'results_{sys.argv[1]}_{sys.argv[6]}.txt', "a")
    h = open(f'feat_{sys.argv[1]}_{sys.argv[6]}.txt', 'a')
    k = open(f'mprobs_{sys.argv[1]}_{sys.argv[6]}.txt', 'a')

    print(f"run {i+1}:")
    with open("main_mf.py") as f:
        exec(f.read())
    a, b, c = best_gen['RMSE'], best_gen['MAE'], best_gen['corr']
    #g.write("{:<35} {:<35} {:<35}\n".format(a, b, c))
    h.write('{}\n'.format(best_gen['features'][best_gen['mprobs'] > 0.3]))
    k.write('{}\n'.format(best_gen['mprobs'][best_gen['mprobs'] > 0.3]))

    f.close()
    #g.close()
    h.close()
    k.close()