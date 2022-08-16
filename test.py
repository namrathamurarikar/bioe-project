import ftplib

import urllib 

'''
session = ftplib.FTP_TLS('ftp.drivehq.com','cloudfilestorageacademic','Offenburg965#')
file = open('test.txt','rb')                  # file to send
session.storbinary('test.txt', file)     # send the file
file.close()                                    # close file and FTP
session.quit()
'''

ftp = ftplib.FTP_TLS("ftp.drivehq.com")
ftp.login("cloudfilestorageacademic", "Offenburg965#")
ftp.prot_p()
'''
file = open("test.txt", "rb")
ftp.storbinary("STOR test.txt", file)
file.close()
ftp.close()
'''
filenames = ftp.nlst()

for filename in filenames:
    with open( filename, 'wb' ) as file :
        ftp.retrbinary('RETR %s' % filename, file.write)

    file.close()
