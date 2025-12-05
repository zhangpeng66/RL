//////////////以下脚本源码在RL/lerobot/src/lerobot/scripts文件夹里面
  - 底座1号 -->肩部旋转部件2号 -->大臂3号 --> 小臂4号 --> 腕部5号 --> 夹爪6号
//查看端口号
python -m lerobot.scripts.lerobot_find_port
sudo chmod 666 /dev/ttyACM0
//连接机械臂的驱动板的 TypeC 到电脑，输入如下指令：
udevadm info -a -n /dev/ttyACM0 | grep serial
//将输出的上面那一行的编码值替换下面ATTRS{serial}=号后面的引号标红内容，保存到 99_so101_serial.rules 文件中
//运行下面的指令，将规则文件写入Ubuntu系统目录，此后将会自动识别左、右机械臂
sudo cp sim_test/configs/99_so101_serial.rules /etc/udev/rules.d/
sudo chmod +x /etc/udev/rules.d/99_so101_serial.rules 


//绑定相机需要以下命令，文件保存在/etc/udev/rules.d/usb_cam.rules
lsusb
v4l2-ctl --list-devices
udevadm info --attribute-walk --name=/dev/video2
udevadm info --attribute-walk --name=/dev/video2 | grep KERNELS

sudo chmod +x usb_cam.rules 

//查看usb相机是否绑定成功
ls -l /dev | grep video



//follow arm
python -m lerobot.scripts.lerobot_setup_motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
python -m lerobot.scripts.lerobot_calibrate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=R20191207
//leader arm
python -m lerobot.scripts.lerobot_setup_motors --teleop.type=so101_leader --teleop.port=/dev/ttyACM0
python -m lerobot.scripts.lerobot_calibrate     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=R20241230

#双臂示教
python -m lerobot.scripts.lerobot_teleoperate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=R20191207     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=R20241230
#查找相机
python -m lerobot.scripts.lerobot_find_cameras

//显示手臂相机
python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 4, 'width': 640, 'height': 480, 'fps': 20}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=R20241230 \
    --display_data=true
//显示两个相机
python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 4, 'width': 640, 'height': 480, 'fps': 20},'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=R20241230 \
    --display_data=true
//更改端口映射之后
python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20},'fixed': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader_left \
    --teleop.id=R20241230 \
    --display_data=true

//录制数据
//--dataset.episode_time_s=20 整个任务的周期，以秒为单位,默认为60，根据完成任务所需时间的长短来设置
//dataset.reset_time_s 执行完任务后场景恢复到初始状态所需的时间，以秒为单位,默认为60，如果场景恢复很快，可以设置短一些，比如5秒。如果场景很复杂，涉及多件物品的复位，则需要设置得长一些
python -m lerobot.scripts.lerobot_record \
    --robot.disable_torque_on_disconnect=true \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20},'fixed': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader_left \
    --teleop.id=R20241230 \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.num_episodes=10 --dataset.episode_time_s=20 \
    --dataset.single_task="Grab the orange ball and put it in the blue box"

//本地回放录制数据
python -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id zp_robot/so101_test \
    --episode-index 40
//处理数据集
export PYTHONPATH=/home/ahpc/RL/lerobot/src
python3 process_dataset.py 

//打印数据集信息
Original dataset: 60 episodes, 19493 frames
Features: ['action', 'observation.state', 'observation.images.handeye', 'observation.images.fixed', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

//推荐新手使用ACT，由 "--policy.type=act" 指定训练的策略为ACT。
//其他可选择的策略：
- act, Action Chunking Transformers
- diffusion, Diffusion Policy  看了bilibi视频说由于unet网络不如transformer
- tdmpc, TDMPC Policy 目前lerobot只支持一个相机
- vqbet, VQ-BeT
- smolvla, SmolVLA
- pi0, A Vision-Language-Action Flow Model for General Robot Control
- pi0fast
- sac
- reward_classifier

//训练数据
export HF_USER=zp_robot

python -m lerobot.scripts.lerobot_train  \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=16 \
  --num_workers=4

//继续训练
python -m lerobot.scripts.lerobot_train \
  --config_path=outputs/train/act_so101_diffusion/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
#带模型训练
python -m lerobot.scripts.lerobot_train  \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model

//推理测试
export HF_USER=zp_robot
cd RL/lerobot/src
rm -rf ~/.cache/huggingface/lerobot/zp_robot/eval_so101

python -m lerobot.scripts.lerobot_record \
    --robot.disable_torque_on_disconnect=true \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20},'fixed': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader_left \
    --teleop.id=R20241230 \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.num_episodes=10 --dataset.episode_time_s=20 \
    --dataset.single_task="Grab the orange ball and put it in the blue box" \
    --policy.path=outputs/checkpoints_act/last/pretrained_model \
    --policy.device=cuda \
    --dataset.repo_id=${HF_USER}/eval_so101 --dataset.push_to_hub=false



// SmolVLA训练
//安装依赖包
pip install -e ".[smolvla]"
//开始训练
export HF_ENDPOINT=https://hf-mirror.com

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=${HF_USER}/so101_test_merged --policy.push_to_hub=false \
  --policy.path=lerobot/smolvla_base --policy.device=cuda \
  --output_dir=outputs/train/smolvla_test \
  --job_name=smolvla_test \
  --batch_size=32 --steps=100000 \
  --wandb.enable=false --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.handeye": "observation.images.camera2"}'
//训练不带基础模型
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=${HF_USER}/so101_test_merged --policy.push_to_hub=false \
  --policy.type=smolvla --policy.device=cuda \
  --output_dir=outputs/train/smolvla_nobasemodel \
  --job_name=smolvla_nobasemodel \
  --batch_size=32 --steps=100000 \
  --wandb.enable=false --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.handeye": "observation.images.camera2"}'

//smolvla 推理测试
export HF_ENDPOINT=https://hf-mirror.com

python -m lerobot.scripts.lerobot_record \
    --robot.disable_torque_on_disconnect=true \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'camera2': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20},'camera1': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader_left \
    --teleop.id=R20241230 \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.num_episodes=10 --dataset.episode_time_s=20 \
    --dataset.single_task="Grab the orange ball and put it in the blue box" \
    --policy.path=outputs/checkpoints_smolvla/last/pretrained_model \
    --policy.device=cuda \
    --dataset.repo_id=${HF_USER}/eval_so101 --dataset.push_to_hub=false
//推理smol_nobasemodel 效果差点
python -m lerobot.scripts.lerobot_record \
    --robot.disable_torque_on_disconnect=true \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20},'fixed': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader_left \
    --teleop.id=R20241230 \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_test \
    --dataset.num_episodes=10 --dataset.episode_time_s=20 \
    --dataset.single_task="Grab the orange ball and put it in the blue box" \
    --policy.path=outputs/checkpoints_smolvla_nobasemodel/last/pretrained_model \
    --policy.device=cuda \
    --dataset.repo_id=${HF_USER}/eval_so101 --dataset.push_to_hub=false 

//训练pi0
//安装依赖包
pip install -e ".[pi]"
//pi0要的paligemma模型，需要授权下载前要登录一下，登入方法如下：
输入python，然后输入：

from huggingface_hub import login
login()

这里输入你hf_开头的huggingface的token，成功后，ctrl+z还是ctrl+d退出
如果有红字提示说要执行git config --global credential.helper store，就执行一下yes

如果遇到了CUDA out of memory，可以尝试把置为True(微调效果待评估)，num_learnable_params由3B降到578M，这样就可以在单张RTX4090(24GB)上微调，此时VRAM占用10950MB，
//开始训练，减少batch_size没有效果 
//erobot/pi0，是action expert 模型，paligemma-3b-mix-224 是pre-trained VLM全量模型需要显存>70G，目前不支持lora需要>22.5G
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=${HF_USER}/so101_test_merged  --policy.push_to_hub=false\
    --policy.type=pi0  --wandb.enable=false\
    --output_dir=outputs/train/pi0_test \
    --job_name=pi0_training \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=100000 \
    --policy.device=cuda \
    --batch_size=1 






export PYTHONPATH=/home/ahpc/RL/percipio
export LD_LIBRARY_PATH=/home/ahpc/RL/percipio
python3 zprobot_find_cameras.py  percipio

//huggingface account
zhangpeng
Username:zp65
secret:@Zpqazwsx65