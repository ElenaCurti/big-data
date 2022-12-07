# Per generare i files splittati da caricare su GitHub:
#split -b 99M -x --additional-suffix=.dat temperatures.dat splitted_temperatures

# Per eseguire il merge:
#cat splitted_temperatures00.dat splitted_temperatures01.dat > temperatures2.dat
