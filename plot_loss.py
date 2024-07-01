import matplotlib.pyplot as plt

LOG_FILE = 'log/log.txt'

val_loss = {'step': [], 'loss':[]}
train_loss = {'step': [], 'loss':[]}

with open(LOG_FILE, 'r') as f:
    file_content = f.readlines()

for line in file_content:
    row = line.split()
    if row[1] == 'val':
        val_loss['step'].append(int(row[0]))
        val_loss['loss'].append(float(row[2]))
    else:
        train_loss['step'].append(int(row[0]))
        train_loss['loss'].append(float(row[2]))

fig, ax = plt.subplots()
ax.plot(train_loss['step'],  train_loss['loss'], c='b', label='train')
ax.plot(val_loss['step'],  val_loss['loss'], c='r', label='val')
ax.set_xlabel(f'step')
ax.set_ylabel(f"loss")
ax.set_ylim([2.8,4.2])
ax.set_title(f"Cross-entropy loss")
plt.legend(loc="upper right")
plt.savefig(f"loss.png",dpi=300)
plt.close()