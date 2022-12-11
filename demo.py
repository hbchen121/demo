
names = []
with open("veri_names_trans.txt", "r") as f:
    names = f.read().strip().split('\n')

query_feats = []
with open("VeRi-chap4-1015-1-query_feats.txt", "r") as f:
    l = f.read().strip().split('\n')
    for i in range(len(l)):
        l[i] = l[i].strip().split(' ')
        for j in range(len(l[i])):
            l[i][j] = float(l[i][j])
        query_feats.append(list(l[i]))

gallery_feats = []
with open("VeRi-chap4-1015-1-gallery_feats.txt", "r") as f:
    l = f.read().strip().split('\n')
    for i in range(len(l)):
        l[i] = l[i].strip().split(' ')
        for j in range(len(l[i])):
            l[i][j] = float(l[i][j])
        gallery_feats.append(list(l[i]))

print(len(names))
print(len(query_feats))
print(len(gallery_feats))
