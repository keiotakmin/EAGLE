import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time  # timeモジュールを追加

# 再現性のための設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# EAGLEオプティマイザ
class EAGLE(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, amsgrad=False, 
                 adaptive_threshold=True, initial_threshold=5e-4):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        amsgrad=amsgrad, adaptive_threshold=adaptive_threshold,
                        threshold=initial_threshold)
        
        super(EAGLE, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(EAGLE, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def get_adaptive_threshold(self, state, grad_norm):
        """勾配ノルムに基づく適応的な閾値の計算"""
        if not hasattr(state, 'grad_norm_history'):
            state['grad_norm_history'] = []
        
        state['grad_norm_history'].append(grad_norm.item())
        
        # 直近の勾配ノルムの履歴を使用（最大10エポック分）
        history = state['grad_norm_history'][-10:]
        
        if len(history) >= 5:
            # 勾配ノルムの変動に基づいて閾値を調整
            grad_std = np.std(history)
            grad_mean = np.mean(history)
            # 変動係数に基づく閾値の調整（変動が大きいほど閾値を大きく）
            cv = grad_std / (grad_mean + 1e-8)
            return max(1e-5, min(1e-2, cv * 5e-3))
        else:
            return state['threshold']
    
    @torch.no_grad()
    def step(self, closure=None):
        """最適化ステップを実行"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # 勾配が存在しない場合はスキップ
                if grad.is_sparse:
                    raise RuntimeError('EAGLE does not support sparse gradients')
                
                # 重み減衰の適用（AdamWスタイル）
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # 状態の取得または初期化
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # 勾配の指数移動平均
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 勾配の二乗の指数移動平均
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # AMSGrad用の最大二次モーメント
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 前回のパラメータと勾配を保存
                    state['prev_param'] = p.data.clone()
                    state['prev_grad'] = torch.zeros_like(p.data)
                    # 閾値の初期化
                    state['threshold'] = group['threshold']
                    # EAGLE更新の使用回数カウンタ
                    state['eagle_count'] = 0
                    state['adam_count'] = 0
                
                # 現在のパラメータと勾配
                curr_param = p.data.clone()
                curr_grad = grad.clone()
                
                # パラメータと勾配の変化量
                delta_param = curr_param - state['prev_param']
                delta_grad = curr_grad - state['prev_grad']
                
                # 勾配ノルムの計算（適応的閾値のため）
                grad_norm = torch.norm(curr_grad)
                
                # 適応的閾値の更新
                if group['adaptive_threshold'] and state['step'] > 0:
                    state['threshold'] = self.get_adaptive_threshold(state, grad_norm)
                
                # Adamのパラメータ
                beta1, beta2 = group['betas']
                
                # 移動平均の更新
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                # 勾配の移動平均の更新
                exp_avg.mul_(beta1).add_(curr_grad, alpha=1 - beta1)
                
                # 勾配の二乗の移動平均の更新
                exp_avg_sq.mul_(beta2).addcmul_(curr_grad, curr_grad, value=1 - beta2)
                
                # AMSGradの場合
                if group['amsgrad']:
                    torch.maximum(state['max_exp_avg_sq'], exp_avg_sq, out=state['max_exp_avg_sq'])
                    denom = (state['max_exp_avg_sq'].sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                
                # バイアス補正
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                
                # Adamステップの計算
                adam_step = exp_avg / denom
                
                # EAGLE更新とAdam更新の条件マスク
                # 条件1: 前回と今回の勾配の符号が同じ
                # 条件2: 勾配の変化と現在の勾配の符号が同じ
                # 条件3: 勾配差の絶対値が閾値より大きい
                condition1 = (state['prev_grad'] * curr_grad >= 0)
                condition2 = (curr_grad * delta_grad >= 0)
                condition3 = (torch.abs(delta_grad) >= state['threshold'])
                
                # 基本マスク: 条件1かつ条件2、または勾配差が小さい場合はAdam
                adam_mask = (condition1 & condition2) | ~condition3
                eagle_mask = ~adam_mask
                
                # 使用回数のカウント
                state['adam_count'] += torch.sum(adam_mask).item()
                state['eagle_count'] += torch.sum(eagle_mask).item()
                
                # Adam更新
                p.data[adam_mask] -= step_size * adam_step[adam_mask]
                
                # EAGLE更新則
                safe_denom = delta_grad.clone()
                # ゼロ除算回避
                safe_denom[torch.abs(safe_denom) < 1e-8] = 1e-8
                p.data[eagle_mask] -= group['lr'] * curr_grad[eagle_mask] * delta_param[eagle_mask] / safe_denom[eagle_mask]
                
                # 現在のパラメータと勾配を保存
                state['prev_param'] = curr_param
                state['prev_grad'] = curr_grad
        
        return loss

# ViT for CIFAR10 モデル定義
class ViTForCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTForCIFAR10, self).__init__()
        # 事前訓練済みのViT-B/16モデルをロード
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # ViTのヘッド部分を修正（CIFAR-10用に10クラス分類に変更）
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, num_classes)
        )
        
        # CIFAR-10の小さな画像に対応するためのアップサンプリング層
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # CIFAR-10の32x32の画像をViTが想定する224x224にアップサンプリング
        x = self.upsample(x)
        return self.vit(x)

# データローダーの設定
def get_cifar10_loaders(batch_size=64, num_workers=2):
    # ViTの前処理設定に合わせる
    # ImageNetの正規化パラメータを使用
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

# 評価関数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # バッチデータをデバイスに転送
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 損失の累積
            total_loss += loss.item()
            
            # 予測結果の保存
            _, predictions = outputs.max(1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # 平均損失
    avg_loss = total_loss / len(test_loader)
    
    # 精度の計算（0-100%スケール）
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    return avg_loss, accuracy

# EAGLEアップデート使用率の計算
def compute_eagle_update_ratio(optimizer):
    """
    オプティマイザからEAGLEアップデートの使用率を計算
    """
    total_eagle_count = 0
    total_base_count = 0  # SGDまたはAdam
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            
            state = optimizer.state[p]
            if len(state) == 0:
                continue
            
            if 'eagle_count' in state and ('sgd_count' in state or 'adam_count' in state):
                total_eagle_count += state['eagle_count']
                if 'sgd_count' in state:
                    total_base_count += state['sgd_count']
                elif 'adam_count' in state:
                    total_base_count += state['adam_count']
    
    # 0除算を防ぐ
    total_updates = total_eagle_count + total_base_count
    if total_updates == 0:
        return 0.0
    
    return total_eagle_count / total_updates

# 訓練関数
def train(model, optimizer, train_loader, test_loader, criterion, epoch, device, 
          results_dict, optimizer_name, log_interval=10):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # エポック開始時間
    epoch_start_time = time.time()
    cumulative_time = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # バッチ開始時間
        batch_start_time = time.time()
        
        # バッチデータをデバイスに転送
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 勾配のリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 逆伝播と最適化
        loss.backward()
        optimizer.step()
        
        # バッチ終了時間
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        cumulative_time += batch_time
        
        # 損失の累積
        total_loss += loss.item()
        
        # 予測結果の保存
        _, predictions = outputs.max(1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        
        # 現在のステップ数を計算（全体のステップ数）
        global_step = epoch * len(train_loader) + batch_idx
        
        # 一定間隔でメトリクスを記録
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
            # 現在のバッチまでの平均損失
            current_loss = total_loss / (batch_idx + 1)
            
            # これまでの予測に基づく精度の計算（0-100%スケール）
            current_acc = accuracy_score(all_labels, all_preds) * 100
            
            # テストも行い、結果を記録
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # 現在の累積時間を記録
            current_time = cumulative_time
            
            # EAGLE系オプティマイザの場合は更新率も計算
            if 'EAGLE' in optimizer_name:
                eagle_ratio = compute_eagle_update_ratio(optimizer)
                results_dict[optimizer_name]['steps'].append(global_step)
                results_dict[optimizer_name]['train_loss'].append(current_loss)
                results_dict[optimizer_name]['train_acc'].append(current_acc)
                results_dict[optimizer_name]['test_loss'].append(test_loss)
                results_dict[optimizer_name]['test_acc'].append(test_acc)
                results_dict[optimizer_name]['eagle_update_ratio'].append(eagle_ratio)
                results_dict[optimizer_name]['time'].append(current_time)
                
                progress_bar.set_postfix({
                    'Loss': current_loss, 
                    'Acc': f'{current_acc:.2f}%', 
                    'EAGLE Ratio': eagle_ratio,
                    'Time': f'{current_time:.2f}s'
                })
            else:
                results_dict[optimizer_name]['steps'].append(global_step)
                results_dict[optimizer_name]['train_loss'].append(current_loss)
                results_dict[optimizer_name]['train_acc'].append(current_acc)
                results_dict[optimizer_name]['test_loss'].append(test_loss)
                results_dict[optimizer_name]['test_acc'].append(test_acc)
                results_dict[optimizer_name]['time'].append(current_time)
                
                progress_bar.set_postfix({
                    'Loss': current_loss, 
                    'Acc': f'{current_acc:.2f}%',
                    'Time': f'{current_time:.2f}s'
                })
    
    # エポックの平均損失
    avg_loss = total_loss / len(train_loader)
    
    # 精度の計算（0-100%スケール）
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    return avg_loss, accuracy

def analyze_convergence_speed(results):
    """
    各オプティマイザの収束速度を比較し、EAGLEが他のオプティマイザの最終的な訓練損失に達するまでのステップ数を計算
    
    Args:
        results: 各オプティマイザの訓練結果を含む辞書
        
    Returns:
        比較分析の結果を含む辞書
    """
    # 分析結果の保存用辞書
    convergence_analysis = {}
    
    # 分析対象のEAGLEオプティマイザ
    eagle_optimizers = ['EAGLE']
    standard_optimizers = ['SGD_Momentum', 'Adam']
    
    # 各EAGLEオプティマイザについて分析
    for eagle_opt in eagle_optimizers:
        if eagle_opt not in results:
            continue
            
        convergence_analysis[eagle_opt] = {}
        
        eagle_steps = results[eagle_opt]['steps']
        eagle_losses = results[eagle_opt]['train_loss']
        
        # 各標準オプティマイザの最終損失に達するまでのステップ数を計算
        for std_opt in standard_optimizers:
            if std_opt not in results:
                continue
                
            # 標準オプティマイザの最終損失
            std_final_loss = results[std_opt]['train_loss'][-1]
            std_final_step = results[std_opt]['steps'][-1]
            
            # EAGLE最終損失
            eagle_final_loss = eagle_losses[-1]
            eagle_final_step = eagle_steps[-1]
            
            # より正確な到達ポイントを特定するシンプルで堅牢なアルゴリズム
            reached_step = None
            reached_loss = None
            
            # 各損失値を調べて、標準オプティマイザの最終損失値以下になった最初のステップを特定
            for i in range(len(eagle_losses)):
                current_loss = eagle_losses[i]
                
                # 標準オプティマイザの最終損失値に到達または下回った場合
                if current_loss <= std_final_loss:
                    reached_step = eagle_steps[i]
                    reached_loss = current_loss
                    
                    # 補間による精度向上（前のステップが存在し、前のステップの損失が閾値より大きい場合）
                    if i > 0 and eagle_losses[i-1] > std_final_loss:
                        prev_loss = eagle_losses[i-1]
                        prev_step = eagle_steps[i-1]
                        
                        # 単純な線形補間（数値的に安定した計算）
                        slope = (current_loss - prev_loss) / (eagle_steps[i] - prev_step)
                        if slope != 0:  # 0除算を避ける
                            # 補間されたステップ（整数に丸める）
                            interpolated_step = int(round(prev_step + (std_final_loss - prev_loss) / slope))
                            
                            # この補間値を整数として使用
                            reached_step = interpolated_step
                            reached_loss = std_final_loss  # 補間点での損失は標準オプティマイザの最終損失と等しい
                    
                    break  # 最初に見つかったポイントで終了
            
            # 結果の保存
            if reached_step is not None:
                # ステップ数の差と速度向上率の計算
                step_diff = std_final_step - reached_step
                speedup_ratio = std_final_step / reached_step if reached_step > 0 else float('inf')
                
                convergence_analysis[eagle_opt][std_opt] = {
                    'standard_final_loss': std_final_loss,
                    'standard_final_step': std_final_step,
                    'eagle_reached_step': reached_step,
                    'step_difference': step_diff,
                    'speedup_ratio': speedup_ratio,
                    'eagle_reached_loss': reached_loss,
                    'eagle_final_loss': eagle_final_loss,
                    'eagle_final_step': eagle_final_step
                }
            else:
                # EAGLEオプティマイザが標準オプティマイザの最終損失に達しなかった場合
                convergence_analysis[eagle_opt][std_opt] = {
                    'standard_final_loss': std_final_loss,
                    'standard_final_step': std_final_step,
                    'eagle_reached_step': None,
                    'step_difference': None,
                    'speedup_ratio': None,
                    'eagle_final_loss': eagle_final_loss,
                    'eagle_final_step': eagle_final_step,
                    'note': "標準オプティマイザの最終損失に達しなかった"
                }
            
            # 追加分析：標準オプティマイザが各ステップで達成した損失をEAGLEが何ステップで達成したか
            step_by_step_analysis = []
            for std_step_idx, std_loss in enumerate(results[std_opt]['train_loss']):
                std_step = results[std_opt]['steps'][std_step_idx]
                # EAGLEがこの損失に達した最初のステップを見つける
                eagle_reached_step = None
                eagle_loss_at_step = None
                
                for eagle_step_idx, eagle_loss in enumerate(eagle_losses):
                    if eagle_loss <= std_loss:
                        eagle_reached_step = eagle_steps[eagle_step_idx]
                        eagle_loss_at_step = eagle_loss
                        break
                
                if eagle_reached_step is not None:
                    # ステップ差と速度向上率
                    step_diff = std_step - eagle_reached_step
                    speedup = std_step / eagle_reached_step if eagle_reached_step > 0 else float('inf')
                    
                    step_by_step_analysis.append({
                        'standard_step': std_step,
                        'standard_loss': std_loss,
                        'eagle_step': eagle_reached_step,
                        'eagle_loss': eagle_loss_at_step,
                        'step_difference': step_diff,
                        'speedup_ratio': speedup
                    })
            
            # 詳細分析の保存 (10ステップごとに)
            convergence_analysis[eagle_opt][f'{std_opt}_detailed'] = step_by_step_analysis[::10]
    
    return convergence_analysis

# オプティマイザの比較実験
def run_optimizer_comparison(seed=42, num_epochs=3, batch_size=64, log_interval=100):
    set_seed(seed)
    
    # CIFAR-10データローダーの準備
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()
    
    # オプティマイザの設定
    optimizers = {
        'SGD_Momentum': lambda m: optim.SGD(
            m.parameters(), 
            lr=0.01,
            momentum=0.9, 
            weight_decay=1e-4
        ),
        'Adam': lambda m: optim.Adam(
            m.parameters(), 
            lr=0.0001, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=1e-4
        ),
        'EAGLE': lambda m: EAGLE(
            m.parameters(), 
            lr=0.0001,
            betas=(0.9, 0.999), 
            weight_decay=1e-4,
            adaptive_threshold=True,
            initial_threshold=5e-4
        )
    }
    
    # 結果の保存用辞書（ステップベース）
    results = {name: {
        'steps': [],  # ステップ数を保存
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'time': [],  # 学習時間を追加
        'eagle_update_ratio': [] if 'EAGLE' in name else None  # EAGLE系オプティマイザのみ追跡
    } for name in optimizers.keys()}
    
    # 各オプティマイザでの訓練
    for name, opt_fn in optimizers.items():
        print(f"\n=== Training with {name} ===")
        
        # ViTモデルの初期化
        model = ViTForCIFAR10(num_classes=10).to(device)
        
        # オプティマイザの初期化
        optimizer = opt_fn(model)
        
        # 訓練ループ
        for epoch in range(num_epochs):
            # 訓練
            train_loss, train_acc = train(
                model, optimizer, train_loader, test_loader, criterion, epoch, device,
                results, name, log_interval=log_interval
            )
            
            # エポック終了時の情報表示
            if 'EAGLE' in name and results[name]['eagle_update_ratio'] is not None:
                last_eagle_ratio = results[name]['eagle_update_ratio'][-1]
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Test Loss: {results[name]['test_loss'][-1]:.4f}, Test Acc: {results[name]['test_acc'][-1]:.2f}%, "
                      f"EAGLE Update Ratio: {last_eagle_ratio:.4f}, "
                      f"Time: {results[name]['time'][-1]:.2f}s")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Test Loss: {results[name]['test_loss'][-1]:.4f}, Test Acc: {results[name]['test_acc'][-1]:.2f}%, "
                      f"Time: {results[name]['time'][-1]:.2f}s")
        
        # モデルの保存
        model_save_path = f'vit_cifar10_{name}_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
    
    return results

# 結果の可視化（ステップベース）
def plot_results(results, num_epochs):
    plt.figure(figsize=(20, 24))
    
    # 訓練損失（左上）
    plt.subplot(3, 2, 1)
    for name, res in results.items():
        plt.plot(res['steps'], res['train_loss'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(-0.02, 0.5)
    plt.grid(True)
    
    # 訓練精度（右上）
    plt.subplot(3, 2, 2)
    for name, res in results.items():
        plt.plot(res['steps'], res['train_acc'], label=name)
    plt.title('Training Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.ylim(68, 101)
    plt.grid(True)
    
    # テスト損失（左中央）
    plt.subplot(3, 2, 3)
    for name, res in results.items():
        plt.plot(res['steps'], res['test_loss'], label=name)
    plt.title('Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # テスト精度（右中央）
    plt.subplot(3, 2, 4)
    for name, res in results.items():
        plt.plot(res['steps'], res['test_acc'], label=name)
    plt.title('Test Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # EAGLE更新率（左下）
    plt.subplot(3, 2, 5)
    for name, res in results.items():
        if 'EAGLE' in name and res['eagle_update_ratio'] is not None:
            plt.plot(res['steps'], res['eagle_update_ratio'], label=name)
    plt.title('EAGLE Update Usage Ratio')
    plt.xlabel('Steps')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    # 学習時間（右下）- F1スコアの代わりに学習時間を表示
    plt.subplot(3, 2, 6)
    for name, res in results.items():
        plt.plot(res['steps'], res['time'], label=name)
    plt.title('Training Time')
    plt.xlabel('Steps')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vit_cifar10_optimizer_comparison_steps.png')
    plt.show()

# 収束速度の可視化
def plot_convergence_analysis(results, convergence_analysis):
    """
    収束速度の分析結果を可視化する
    """
    eagle_opts = [opt for opt in convergence_analysis.keys()]
    std_opts = ['SGD_Momentum', 'Adam']
    
    if not eagle_opts:
        print("収束分析に十分なデータがありません。")
        return
    
    # 各EAGLEオプティマイザについて、標準オプティマイザとの比較グラフを作成
    for eagle_opt in eagle_opts:
        plt.figure(figsize=(20, 8))
        
        # グラフ1: 訓練損失の比較と到達ポイントのマーキング
        plt.subplot(1, 2, 1)
        
        # すべてのオプティマイザの損失をプロット
        for name, res in results.items():
            plt.plot(res['steps'], res['train_loss'], label=name)
            
        # 標準オプティマイザの最終損失値に到達したEAGLEの点をマーク
        colors = {'SGD_Momentum': 'r', 'Adam': 'g'}
        markers = {'SGD_Momentum': 'o', 'Adam': 's'}
        
        for std_opt in std_opts:
            if std_opt in convergence_analysis[eagle_opt]:
                analysis = convergence_analysis[eagle_opt][std_opt]
                
                if analysis['eagle_reached_step'] is not None:
                    # 標準オプティマイザの最終点
                    plt.scatter([analysis['standard_final_step']], [analysis['standard_final_loss']], 
                              c=colors[std_opt], marker=markers[std_opt], s=150, 
                              label=f"{std_opt} Final Loss")
                    
                    # EAGLEの到達点
                    plt.scatter([analysis['eagle_reached_step']], [analysis['eagle_reached_loss']], 
                              c='blue', marker='x', s=150, 
                              label=f"{eagle_opt} reaches {std_opt} final loss")
                    
                    # 二点間を線で結ぶ
                    plt.plot([analysis['eagle_reached_step'], analysis['standard_final_step']], 
                           [analysis['eagle_reached_loss'], analysis['standard_final_loss']], 
                           'k--', alpha=0.5)
        
        plt.title(f'Training Loss Comparison with {eagle_opt}')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.ylim(-0.02, 0.5)
        
        # グラフ2: 詳細な収束速度比較（ステップごとの速度向上率）
        plt.subplot(1, 2, 2)
        
        for std_opt in std_opts:
            detailed_key = f'{std_opt}_detailed'
            if detailed_key in convergence_analysis[eagle_opt]:
                detailed = convergence_analysis[eagle_opt][detailed_key]
                
                if detailed:
                    std_steps = [item['standard_step'] for item in detailed]
                    speedups = [item['speedup_ratio'] for item in detailed]
                    
                    plt.plot(std_steps, speedups, 'o-', label=f'Speedup vs {std_opt}')
        
        plt.title(f'Convergence Speedup of {eagle_opt} vs Standard Optimizers')
        plt.xlabel('Standard Optimizer Steps')
        plt.ylabel('Speedup Ratio (Standard Steps / EAGLE Steps)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{eagle_opt}_convergence_analysis.png')
        plt.show()

# メイン実行関数（収束分析の追加）
def main():
    print("Starting Optimizer Comparison Experiment on ViT-B/16 with CIFAR-10")
    print("Comparing Three Optimizers: SGD with Momentum, Adam, and EAGLE")
    
    # 実験の実行
    num_epochs = 7
    batch_size = 64  # メモリ制約に合わせて調整
    log_interval = 50  # 50バッチごとにメトリクスを記録
    
    results = run_optimizer_comparison(
        seed=42, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        log_interval=log_interval
    )
    
    # 収束速度の分析を実行
    convergence_analysis = analyze_convergence_speed(results)
    
    # 結果の可視化
    plot_results(results, num_epochs)
    
    # 収束速度の可視化
    plot_convergence_analysis(results, convergence_analysis)
    
    # 収束分析の結果表示
    print("\n=== Convergence Speed Analysis ===")
    for eagle_opt, analyses in convergence_analysis.items():
        print(f"\n{eagle_opt} Convergence Analysis:")
        print("-" * 80)
        
        for std_opt, analysis in analyses.items():
            if '_detailed' not in std_opt:  # 詳細分析はスキップ
                print(f"\nComparison with {std_opt}:")
                
                if analysis['eagle_reached_step'] is not None:
                    print(f"  {std_opt} final loss: {analysis['standard_final_loss']:.6f} at step {analysis['standard_final_step']}")
                    print(f"  {eagle_opt} reached this loss at step {analysis['eagle_reached_step']}")
                    print(f"  Step difference: {analysis['step_difference']} steps")
                    print(f"  Speedup ratio: {analysis['speedup_ratio']:.2f}x faster")
                else:
                    print(f"  {std_opt} final loss: {analysis['standard_final_loss']:.6f}")
                    print(f"  {eagle_opt} did not reach this loss value within the training period")
                    print(f"  {eagle_opt} final loss: {analysis['eagle_final_loss']:.6f}")
    
    # 最終結果の表示
    print("\nFinal Results:")
    print("-" * 120)
    print(f"{'Optimizer':<25} {'Train Loss':>12} {'Train Acc':>12} {'Test Loss':>12} {'Test Acc':>12} {'EAGLE Usage':>12} {'Time':>12}")
    print("-" * 120)
    
    for name, res in results.items():
        if 'EAGLE' in name and res['eagle_update_ratio'] is not None:
            eagle_usage = f"{res['eagle_update_ratio'][-1]*100:.2f}%"
        else:
            eagle_usage = "N/A"
            
        print(f"{name:<25} {res['train_loss'][-1]:>12.4f} {res['train_acc'][-1]:>11.2f}% "
              f"{res['test_loss'][-1]:>11.4f} {res['test_acc'][-1]:>11.2f}% {eagle_usage:>12} {res['time'][-1]:>11.2f}s")
        
    # 最良性能の比較
    print("\nBest Test Performance Analysis:")
    print("-" * 80)
    print(f"{'Optimizer':<25} {'Best Test Acc':>15} {'Step Reached':>15} {'Time Taken':>15}")
    print("-" * 80)
    
    for name, res in results.items():
        best_acc = max(res['test_acc'])
        best_step_idx = res['test_acc'].index(best_acc)
        best_step = res['steps'][best_step_idx]
        best_time = res['time'][best_step_idx]
        print(f"{name:<25} {best_acc:>14.2f}% {best_step:>15} {best_time:>14.2f}s")
        
    # EAGLE更新率の分析
    print("\nEAGLE Update Usage Analysis:")
    print("-" * 80)
    print(f"{'Optimizer':<25} {'Avg Usage':>15} {'Min Usage':>15} {'Max Usage':>15}")
    print("-" * 80)
    
    for name, res in results.items():
        if 'EAGLE' in name and res['eagle_update_ratio'] is not None:
            avg_usage = np.mean(res['eagle_update_ratio']) * 100
            min_usage = min(res['eagle_update_ratio']) * 100
            max_usage = max(res['eagle_update_ratio']) * 100
            print(f"{name:<25} {avg_usage:>14.2f}% {min_usage:>14.2f}% {max_usage:>14.2f}%")
    
    # 時間効率の分析
    print("\nTime Efficiency Analysis:")
    print("-" * 80)
    print(f"{'Optimizer':<25} {'Total Time':>15} {'Time per 50 steps':>20}")
    print("-" * 80)
    
    for name, res in results.items():
        total_time = res['time'][-1]
        total_steps = res['steps'][-1]
        time_per_50_steps = (total_time / total_steps) * 50 if total_steps > 0 else 0
        print(f"{name:<25} {total_time:>14.2f}s {time_per_50_steps:>19.2f}s")

if __name__ == "__main__":
    main()