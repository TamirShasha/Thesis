import matplotlib.pyplot as plt
import matplotlib.patches as patches

image_path = r'C:\Users\tamir\Desktop\New folder\005_mrc.png'
loc, size = (428, 243), 55

img = plt.imread(image_path)

fig, ax = plt.subplots(figsize=(8, 8))

rect = patches.Rectangle(loc, size, size, color='red', fill=None)
ax.add_patch(rect)

plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.savefig(r'C:\Users\tamir\Desktop\New folder\005_mrc_modified.png')
plt.show()
