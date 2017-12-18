#!/bin/sh
gzserver ../worlds/arm_world.world &
echo "#!/bin/sh\nkill -9 $!\nrm killserver.sh" > killserver.sh
chmod +x killserver.sh
sleep 1
gzclient &
