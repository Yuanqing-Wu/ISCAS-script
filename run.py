import subprocess

# classifier = 's-ns_64x64'
# cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
# subprocess.Popen(cmd, shell=True)

classifier = 's-ns_32x32'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

classifier = 's-ns_16x16'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

classifier = 's-ns_32x16'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

classifier = 's-ns_32x8'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

classifier = 's-ns_16x8'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

classifier = 's-ns_8x8'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

# classifier = 'hs-vs_32x32'
# cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
# subprocess.Popen(cmd, shell=True)

# classifier = 'hs-vs_32x16'
# cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
# subprocess.Popen(cmd, shell=True)

# classifier = 'hs-vs_32x8'
# cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
# subprocess.Popen(cmd, shell=True)

classifier = 'hs-vs_16x16'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

classifier = 'hs-vs_16x8'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)

classifier = 'hs-vs_8x8'
cmd = 'python hh.py --classifier ' + classifier + ' > log/' + classifier + '.txt'
subprocess.Popen(cmd, shell=True)