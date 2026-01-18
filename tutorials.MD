//////////////以下脚本源码在RL/lerobot/src/lerobot/scripts文件夹里面
  - 底座1号 -->肩部旋转部件2号 -->大臂3号 --> 小臂4号 --> 腕部5号 --> 夹爪6号
//查看端口号
```shell
python -m lerobot.scripts.lerobot_find_port
sudo chmod 666 /dev/ttyACM0
```
//连接机械臂的驱动板的 TypeC 到电脑，输入如下指令：
```shell
udevadm info -a -n /dev/ttyACM0 | grep serial
```
//将输出的上面那一行的编码值替换下面ATTRS{serial}=号后面的引号标红内容，保存到 99_so101_serial.rules 文件中
//运行下面的指令，将规则文件写入Ubuntu系统目录，此后将会自动识别左、右机械臂
```shell
sudo cp sim_test/configs/99_so101_serial.rules /etc/udev/rules.d/
sudo chmod +x /etc/udev/rules.d/99_so101_serial.rules 
```

//绑定相机需要以下命令，文件保存在/etc/udev/rules.d/usb_cam.rules
```shell
lsusb
v4l2-ctl --list-devices
udevadm info --attribute-walk --name=/dev/video2
udevadm info --attribute-walk --name=/dev/video2 | grep KERNELS

sudo chmod +x usb_cam.rules 
```
//查看usb相机是否绑定成功
```shell
ls -l /dev | grep video
```


//follow arm
```shell
python -m lerobot.scripts.lerobot_setup_motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
python -m lerobot.scripts.lerobot_calibrate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=R20191207
```
//leader arm
```shell
python -m lerobot.scripts.lerobot_setup_motors --teleop.type=so101_leader --teleop.port=/dev/ttyACM0
python -m lerobot.scripts.lerobot_calibrate     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=R20241230
```
#双臂示教
```shell
python -m lerobot.scripts.lerobot_teleoperate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=R20191207     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=R20241230
```
#查找相机
```shell
python -m lerobot.scripts.lerobot_find_cameras
```
//显示手臂相机
```shell
python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 4, 'width': 640, 'height': 480, 'fps': 20}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=R20241230 \
    --display_data=true
```
//显示两个相机
```shell
python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': 4, 'width': 640, 'height': 480, 'fps': 20},'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=R20241230 \
    --display_data=true
```
//更改端口映射之后
```shell
python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'handeye': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20},'fixed': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader_left \
    --teleop.id=R20241230 \
    --display_data=true
```
//录制数据(任务名字不能混淆,需要统一)
//--dataset.episode_time_s=50 整个任务的周期，以秒为单位,默认为60，根据完成任务所需时间的长短来设置
//dataset.reset_time_s 执行完任务后场景恢复到初始状态所需的时间，以秒为单位,默认为60，如果场景恢复很快，可以设置短一些，比如5秒。如果场景很复杂，涉及多件物品的复位，则需要设置得长一些
//任务1:  Pick up the blue duck toy and put it in the blue box
//任务2： Pick up the orange ball and put it in the blue box
```shell
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
    --dataset.num_episodes=10 --dataset.episode_time_s=50 \
    --dataset.single_task="Pick up the orange ball and put it in the blue box"
```
//本地回放录制数据
```shell
python -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id zp_robot/so101_test \
    --episode-index 40
```
//处理数据集
```shell
export PYTHONPATH=/home/ahpc/RL/lerobot/src
python3 process_dataset.py 
```
//打印数据集信息
Original dataset: 60 episodes, 19493 frames
Features: ['action', 'observation.state', 'observation.images.handeye', 'observation.images.fixed', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

//推荐新手使用ACT，由 "--policy.type=act" 指定训练的策略为ACT。
//其他可选择的策略：
- act, Action Chunking Transformers
- diffusion, Diffusion Policy  看了bilibi视频说由于unet网络不如transformer
- tdmpc, TDMPC Policy 目前lerobot只支持一个相机
- vqbet, VQ-BeT
- smolvla, SmolVLA  SmolVLA依赖多摄像头观测（如顶部、手腕、侧面视角）。若实际摄像头数量少于模型预期（例如模型需 3 个摄像头但实体仅安装 2 个），需通过占位补齐空缺。
- pi0, A Vision-Language-Action Flow Model for General Robot Control
- pi0fast
- sac
- reward_classifier

//训练数据
```shell
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
```

//训练diffusion需要更改配置(https://github.com/huggingface/lerobot/issues/2506)
```shell
python -m lerobot.scripts.lerobot_train  \
  --dataset.repo_id=${HF_USER}//so101_grab_ball_merged1 \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_so101_test \
  --job_name=diffusion_so101_test \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.num_inference_steps=10 \
  --policy.drop_n_last_frames=2 \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.vision_backbone=resnet34 \
  --policy.crop_shape=null \
  --dataset.image_transforms.enable=true \
  --wandb.enable=false \
  --batch_size=1 \
  --num_workers=4
```
//继续训练
```shell
python -m lerobot.scripts.lerobot_train \
  --config_path=outputs/train/act_so101_diffusion/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```
#带模型训练
```shell
python -m lerobot.scripts.lerobot_train  \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model
```
//推理测试
```shell
export HF_USER=zp_robot
cd RL/lerobot/src
rm -rf ~/.cache/huggingface/lerobot/zp_robot/eval_so101
//删除遥操作控制效果很差，后面还是加入了
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
    --dataset.num_episodes=10 --dataset.episode_time_s=50 \
    --dataset.single_task="Pick up the orange ball and put it in the blue box" \
    --policy.path=outputs/act/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --dataset.repo_id=${HF_USER}/eval_so101 --dataset.push_to_hub=false
```


// SmolVLA训练
//安装依赖包
```shell
pip install -e ".[smolvla]"
export HF_ENDPOINT=https://hf-mirror.com
```
//主要使用smolvla基础模型需要按照他的摄像头分布camera1=top,camera2=wrist,camera3=side 
```shell
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=${HF_USER}/so101_test_merged --policy.push_to_hub=false \
  --policy.path=lerobot/smolvla_base --policy.device=cuda \
  --policy.type=smolvla \
  --output_dir=outputs/train/smolvla_test \
  --job_name=smolvla_test \
  --batch_size=32 --steps=100000 \
  --wandb.enable=false --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.handeye": "observation.images.camera2"}'
```
//训练不带基础模型，经过测试效果比较差，原因不详
```shell
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=${HF_USER}/so101_test_merged --policy.push_to_hub=false \
  --policy.type=smolvla --policy.device=cuda \
  --output_dir=outputs/train/smolvla_nobasemodel \
  --job_name=smolvla_nobasemodel \
  --batch_size=32 --steps=100000 \
  --wandb.enable=false --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.handeye": "observation.images.camera2"}'
```
//smolvla多任务训练，利用任务A模型训练任务B,修改config_path 打印显示'pretrained_path': 'lerobot/smolvla_base',
```shell
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=${HF_USER}/so101_dark_merged --policy.push_to_hub=false \
  --policy.type=smolvla --policy.device=cuda \
  --output_dir=outputs/train/smolvla_multitask \
  --job_name=smolvla_multitask \
  --batch_size=32 --steps=100000 \ 
  --config_path=outputs/train/smolvla_test/checkpoints/last/pretrained_model \
  --wandb.enable=false --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.handeye": "observation.images.camera2"}'
```
//smolvla多任务训练，利用任务A模型训练任务B,修改policy.path,打印显示'pretrained_path': 'outputs/train/smolvla_test/checkpoints/last/pretrained_model',
```shell
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=${HF_USER}/so101_dark_merged --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=outputs/train/smolvla_multitask_selfpolicy \
  --job_name=smolvla_multitask_selfpolicy \
  --batch_size=32 --steps=100000 \
  --policy.path=outputs/train/smolvla_test/checkpoints/last/pretrained_model \
  --wandb.enable=false --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.handeye": "observation.images.camera2"}'
```
//总结：多任务训练使用config_path设置模型路径在ball和dark没有效果，正确使用policy.path，录数据的时候任务名字一定要表达好--dataset.single_task
//smolvla 推理测试
export HF_ENDPOINT=https://hf-mirror.com
//不加teleop遥操作效果很差
```shell
python -m lerobot.scripts.lerobot_record \
    --robot.disable_torque_on_disconnect=true \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'camera1': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30},'camera2': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader_left \
    --teleop.id=R20241230 \
    --display_data=true \
    --dataset.num_episodes=10 --dataset.episode_time_s=50 \
    --dataset.single_task="Pick up the orange ball and put it in the blue box" \
    --policy.path=outputs/smolvla_grab_ball1/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --dataset.repo_id=${HF_USER}/eval_so101 --dataset.push_to_hub=false
```
//推理smol_nobasemodel 效果差点
```shell
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
    --dataset.num_episodes=10 --dataset.episode_time_s=50 \
    --dataset.single_task="Pick up the orange ball and put it in the blue box" \
    --policy.path=outputs/checkpoints_smolvla_nobasemodel/last/pretrained_model \
    --policy.device=cuda \
    --dataset.repo_id=${HF_USER}/eval_so101 --dataset.push_to_hub=false 
```
//异步推理
//运行服务端 需要修改频率fps
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=25 \
     --inference_latency=0.040 \
     --obs_queue_timeout=1
```
//运行客户端,在模型的配置文件config.json和train_config.json中需要注明训练的数据集的repo_id==>"repo_id": "zp_robot/so101_grab_ball_merged1",
```shell
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'camera1': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30},'camera2': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20}}" \
    --task="Pick up the orange ball and put it in the blue box" \
    --pretrained_name_or_path=outputs/smolvla_grab_ball1/checkpoints/last/pretrained_model \
    --policy_type=smolvla \
    --actions_per_chunk=20 \
    --chunk_size_threshold=0.5 \
    --policy_device=cuda \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
//Real-Time Chunking(RTC)推理 
//execution_horizon：与前一个数据块保持一致所需的时间步数。数值越高，过渡越平滑，但响应速度可能越慢。 
//max_guidance_weight：与前一个数据块保持一致性的程度。这是一个超参数，可以进行调整以平衡转换的平滑度和策略的响应速度。对于 10 步流匹配（SmolVLA、Pi0、Pi0.5），10.0 是一个最优值。
```shell
//source .venv/bin/activate
uv run examples/rtc/eval_with_real_robot.py \
    --policy.path=src/outputs/smolvla_grab_ball1/checkpoints/last/pretrained_model  \
    --policy.device=cuda \
    --rtc.enabled=true \
    --rtc.execution_horizon=20 \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower_left \
    --robot.id=R20191207 \
    --robot.cameras="{ 'camera1': {'type': 'opencv', 'index_or_path': /dev/fixed_camera, 'width': 640, 'height': 480, 'fps': 30},'camera2': {'type': 'opencv', 'index_or_path': /dev/hand_camera, 'width': 640, 'height': 480, 'fps': 20}}" \
    --task="Pick up the orange ball and put it in the blue box" \
    --duration=120   
```
//训练pi0
//安装依赖包
pip install -e ".[pi]"
//pi0要的paligemma模型，需要授权下载前要登录一下，登入方法如下：
输入python，然后输入：
```shell
from huggingface_hub import login
login()
```
这里输入你hf_开头的huggingface的token，成功后，ctrl+z还是ctrl+d退出
如果有红字提示说要执行git config --global credential.helper store，就执行一下yes

如果遇到了CUDA out of memory，可以尝试把置为True(微调效果待评估)，num_learnable_params由3B降到578M，这样就可以在单张RTX4090(24GB)上微调，此时VRAM占用10950MB，
//开始训练，减少batch_size没有效果 
//erobot/pi0，是action expert 模型，paligemma-3b-mix-224 是pre-trained VLM全量模型需要显存>70G，目前不支持lora需要>22.5G
```shell
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
```
hil-serl录数据
```shell
python -m lerobot.rl.gym_manipulator --config_path /media/zp/Tiger/AHPC/RL/sim_test/configs/gym_hil_env.json
```
hil-serl训练
```shell
python -m lerobot.rl.actor --config_path /home/ahpc/RL/sim_test/configs/train_gym_hil_env.json
python -m lerobot.rl.learner --config_path /home/ahpc/RL/sim_test/configs/train_gym_hil_env.json
```
寻找图漾相机
```shell
export PYTHONPATH=/home/ahpc/RL/percipio
export LD_LIBRARY_PATH=/home/ahpc/RL/percipio
python3 zprobot_find_cameras.py  percipio
```
//huggingface account
zhangpeng
Username:zp65
secret:@Zpqazwsx65

日志文件里面是debug等级的


如果是首次使用蓝牙配对无线控制器与受支持的设备，请打开配对模式。
确保控制器上的玩家指示灯已熄灭。
如果玩家指示灯亮起，请长按PS键直到指示灯熄灭。如果控制器上已连接USB设备，请断开其连接。 
在长按“分享”键的同时，长按PS键，直到光条闪烁。
启用设备上的蓝牙，然后从蓝牙设备列表中选择控制器。配对完成后，光条会闪烁，随后玩家指示灯亮起。
