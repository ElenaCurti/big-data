def prova_fun():
    print("ok")

continuare = True
while(continuare):
    print ('1 -- Option 1' )
    print ('2 -- Option 2' )
    print ('3 -- Option 3' )
    print ('0 -- Exit' )
    option = ''
    try:
        option = input('Enter your choice: ')
    except:
        print('Wrong input. Please enter a number ...')
    #Check what choice was entered and act accordingly
    if option == "0":
        print('Thanks message before exiting')
        continuare = False
    if option == "1":
        print("Hai scelto: 1")
    elif option == "2":
        print("Hai scelto: 2")
    elif option == "Hai scelto: 3":
        print("3")
    else:
        print('Invalid option. Please enter a number between 1 and 4.')