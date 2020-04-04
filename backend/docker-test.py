import docker
from ast import literal_eval
client = docker.from_env()
container = client.containers.run("project", detach=True)
print(container)
container = client.containers.get(container.id)
exit_code, output = container.exec_run("/root/project/bin/python predict.py")
# try catch!
prediction_dict = output.split("\n")[-2]
prediction = literal_eval(prediction_dict)
print(prediction)