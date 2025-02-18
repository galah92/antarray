import subprocess as sp


def main():
    image_name = "openems-image"
    # sim_path = "Simple_Patch_Antenna.py"
    sim_path = "Patch_Antenna_Array.py"

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
