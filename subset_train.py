data_size_plus_header = 1000001
train_dir = 'train'
subset_train_dir = 'sub_train.txt'

fullfile = open(train_dir, 'r')
subfile = open(subset_train_dir,'w')

for i in range(data_size_plus_header):
    subfile.write(fullfile.readline())

fullfile.close()
subfile.close()
