import subprocess

def get_external_ip():
    # The command you want to execute
    cmd = 'curl'
    
    # Execute the command to get external IP using ifconfig.me
    process = subprocess.Popen([cmd, '-s', 'ifconfig.me'], stdout=subprocess.PIPE, text=True)
    
    # Get the output as a string
    myIP, _ = process.communicate()
    
    # Remove any leading or trailing whitespace characters
    myIP = myIP.strip()
    
    return myIP

#if __name__ == '__main__':
#    myIP = get_external_ip()
#    print("My IP address is:", myIP)
