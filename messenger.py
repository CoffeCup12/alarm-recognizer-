
import smtplib

class messenger():
    def __init__(self):

        self.carriers  = {
	'gmail':'@gmail.com',
	'osu':' @buckeyemail.osu.edu',
    }
    
    def generate_message(self, idx):
        message = ""
        match idx:
            case 0:
                message = "A danger alarm"
            case 1:
                message = "A fire alarm"
            case 2:
                message = "A gas alarm"
            case 3:
                message = "not an alarm"
            case 4:
                message = "A tusunami alarm"

        return "there is a " + message + " at your location, please be careful!!!"


    def send(self, idx):

        email = 'youremail'
        password = 'your password' #'lxwx biuk iglk qjrl'
        receiver = 'reciveremail'

        address = receiver.format(self.carriers['osu'])
        auth = (email.format(self.carriers['gmail']), password)

        # Establish a secure session with gmail's outgoing SMTP server 
        server = smtplib.SMTP( "smtp.gmail.com", 587 )
        server.starttls()
        server.login(auth[0], auth[1])

        # Send  email
        server.sendmail( auth[0], address, self.generate_message(idx))

        
