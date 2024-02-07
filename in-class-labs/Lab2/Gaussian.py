# pull an information for a text file

# read the numbers, ignore strings and negative numbers, and calculate the corresponding mean and std for breeds.
# write a new results.txt file for the output. 

file_names = []
for fn in file_names :
    f = open(fn)
    text = f.read()
    pieces = text.split(", ")