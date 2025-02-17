import subprocess as sp


def main():
    print("Hello from openems-docker!")
    IMAGE_NAME = "openems-image"
    cmd = f"""
	docker run -it --rm \
		-e DISPLAY=host.docker.internal:0 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./src:/app/ \
		-v /tmp:/tmp/ \
		{IMAGE_NAME} \
		python3 /app/Simple_Patch_Antenna.py
	"""
    sp.run(cmd, shell=True)


if __name__ == "__main__":
    main()
