from os import walk

f = []
for (dirpath, dirnames2, filenames) in walk('../val'):
    f.extend(filenames)
    break
dirnames2.sort()
with open('../annot.txt', 'w') as fp:
    for item in dirnames2:
        f = []
        for (dirpath, dirnames, filenames) in walk(f'../val/{item}'):
            f.extend(filenames)
            break
        for f in filenames:
            fp.write(item + '/' + f + ' ' + str(dirnames2.index(item)) + '\n')
