import subprocess
import os
try:
    command = "cp wav.scp ../../v13_parliament_v3/test_data"
    subprocess.check_call(command, shell=True)

    # os.chdir("../../v13_parliament_v3")
    command = "../../v13_parliament_v3/run_prod.sh"
    subprocess.check_call(command, shell=True)

    with open("../../v13_parliament_v3/exp/result_prod.txt") as f:
      text = f.read()
      print(text)
except subprocess.CalledProcessError as exc:
  print("Status : FAIL", exc.returncode, exc.output)
  print("CallProcessError")
  # break # handle errors in the called executable
except OSError:
  print("OSERROR")
except FileNotFoundError:
  print("FileNotFoundError")