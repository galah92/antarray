import subprocess as sp


def main():
    image_name = "openems-image"
    # sim_path = "single_antenna.py"
    sim_path = "antenna_array.py"
    # sim_path = "cst.py"

    cmd = f"""
	docker run -it --rm \
		-e DISPLAY=host.docker.internal:0 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./src:/app/ \
		-v /tmp:/tmp/ \
		{image_name} \
		python3 /app/{sim_path}
	"""
    sp.run(cmd, shell=True)


if __name__ == "__main__":
    main()
