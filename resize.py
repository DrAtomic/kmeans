from PIL import Image
scale = int(input("How many times larger do you want to scale the 8x8 pixel grids?\
    \nExample: '2' doubles the size.\n"))
s = 8*scale

for center in range(10):
  name = 'experiment1_center_' + str(center) + '.png'
  original_image = Image.open(name, 'r')
  larger_image = original_image.resize((s,s))
  name2 = '%dx%d_%s' % (s,s,name)
  larger_image.save(name2)
  
for center in range(30):
  name = 'experiment2_center_' + str(center) + '.png'
  original_image = Image.open(name, 'r')
  larger_image = original_image.resize((s,s))
  name2 = '%dx%d_%s' % (s,s,name)
  larger_image.save(name2)
