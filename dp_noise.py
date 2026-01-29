import torch
import math
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
import numpy as np


def sample_noise_Gaussian(d_shape, noise_stddev, device="cpu"):
    noise = torch.normal(mean=0., std=float(noise_stddev), size=d_shape, device=device)
    return noise

def sample_noise_Chi(d_shape, eta, device="cpu"):
    n_dim = d_shape[-1]
    alpha = torch.ones(d_shape) * n_dim
    beta = torch.ones(d_shape) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample()
    # v_lst = -2 * torch.rand(d_shape) + 1
    v_lst = torch.randn(d_shape)
    v_lst = v_lst / torch.norm(v_lst, dim=-1, keepdim=True)
    noise = l_lst * v_lst
    noise = noise.to(device)
    return noise

def quantize_tensor_round(tensor, num_bits):
    # min_val, max_val = tensor.min(dim=2, keepdim=True)[0], tensor.max(dim=2, keepdim=True)[0]
    min_val, max_val = torch.min(tensor), torch.max(tensor)
    q_levels = 2 ** num_bits

    scale = (max_val - min_val) / (q_levels - 1)

    quantized = ((tensor - min_val) / scale).round().clamp(0, q_levels - 1)
    quantized_tensor = quantized * scale + min_val

    return quantized_tensor


def quantize_tensor(tensor, num_bits):
    """
    随机量化函数 - 实现无偏的随机量化 (Stochastic Quantization)
    
    Args:
        tensor: 输入张量
        num_bits: 量化位数
    
    Returns:
        quantized_tensor: 随机量化后的张量
    """
    # 找到输入张量的最小值和最大值
    # min_val, max_val = tensor.min(dim=2, keepdim=True)[0], tensor.max(dim=2, keepdim=True)[0]
    min_val, max_val = torch.min(tensor), torch.max(tensor)
    # 计算量化级别的数量
    q_levels = 2 ** num_bits

    # 计算缩放比例
    scale = (max_val - min_val) / (q_levels - 1)
    
    # 避免除零错误
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # 将连续值映射到 [0, q_levels-1] 范围
    normalized = (tensor - min_val) / scale
    
    # 随机量化：基于小数部分进行概率性舍入
    floor_vals = torch.floor(normalized)
    frac_vals = normalized - floor_vals
    
    # 生成随机数，如果随机数小于小数部分，则向上舍入，否则向下舍入
    random_vals = torch.rand_like(frac_vals)
    quantized = torch.where(random_vals < frac_vals, 
                           floor_vals + 1, 
                           floor_vals)
    
    # 确保量化值在有效范围内
    quantized = quantized.clamp(0, q_levels - 1)
    
    # 还原为原始范围的浮点数
    quantized_tensor = quantized * scale + min_val

    return quantized_tensor


def get_noise_embeddings(init_emb, args, model, ):
    '''
    server compute the embeddings given to the client
    '''

    norm_origin = 1.2
    initial_embeddings = init_emb.clone().detach()
    noises = None
    if args.noise_mechanism == "Gaussian":
        init_emb = torch.clamp(init_emb, min=-args.clip_bound, max=args.clip_bound)
        noise_std = 2*args.clip_bound/args.mu*math.sqrt(init_emb.shape[-1])
        # print(f"noise_std: {noise_std}")
        # print('shape of init_emb:', init_emb.shape)
        noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
        ####sparsity
        random_variable = torch.rand_like(init_emb)
        noises = torch.where(random_variable <= 1 - args.sparsity, -init_emb, noises)

    elif args.noise_mechanism == "ChiDP":
        eta = args.eta
        # print('shape of init_emb:', init_emb.shape)
        noises = sample_noise_Chi(init_emb.shape, eta, args.device)
        ####sparsity
        random_variable = torch.rand_like(init_emb)
        noises = torch.where(random_variable <= 1 - args.sparsity, -init_emb, noises)
    elif args.noise_mechanism == "Gaussian_binary":
        encoder_list = []
        all_norms = torch.norm(init_emb, p=2, dim=-1, keepdim=True)  # keepdim用于广播
        
        scaling_factor = torch.clamp(args.clip_bound / all_norms, max=1.0)
        init_emb = init_emb * scaling_factor
        # noise_std = 2*args.clip_bound/args.mu
        for i in range(args.dp_rounds):
            mu_ = math.sqrt(args.mu ** 2 / args.dp_rounds)
            noise_std = 2*args.clip_bound/mu_
            noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
            encoded = init_emb + noises
            sign_noises = torch.sign(encoded)
            encoder_list.append(sign_noises)
        stacked_tensors = torch.stack(encoder_list)
        encoded_tensor = torch.mean(stacked_tensors, dim=0)
        noises = encoded_tensor - init_emb

    elif args.noise_mechanism == "Ternary":
        encoder_list = []
        init_emb = torch.clamp(init_emb, min=-args.clip_bound, max=args.clip_bound)
        args.dp_rounds=2**(args.quant_level) -1
        for i in range(args.dp_rounds):
            mu_ = math.sqrt(args.mu ** 2 / args.dp_rounds)/math.sqrt(init_emb.shape[-1])

            A = math.sqrt(args.sparsity * (4 / mu_ ** 2 + 1) * args.clip_bound ** 2)  # 此处看一下init_emb有没有batch维度
            B = A / args.sparsity
            random_variable = torch.rand_like(init_emb)
            ones_tensor = B * torch.ones_like(init_emb)
            zeros_tensor = torch.zeros_like(init_emb)
            encoded_tensor = torch.where(random_variable <= (1 / 2 + init_emb / (2 * A)), ones_tensor, -ones_tensor)
            random_variable = torch.rand_like(encoded_tensor)
            encoded_tensor = torch.where(random_variable <= 1 - A / B, zeros_tensor, encoded_tensor)
            encoder_list.append(encoded_tensor)

        stacked_tensors = torch.stack(encoder_list)
        encoded_tensor = torch.mean(stacked_tensors, dim=0)

        noises = encoded_tensor - init_emb
    if noises is not None:
        
        noise_init_emb = init_emb+noises
        # print('pure noise')
    else:
        noise_init_emb = init_emb

    if args.quant_level != 32 and args.noise_mechanism != "Ternary" and args.noise_mechanism != "Gaussian_binary":
        # init_emb = quantize_tensor(init_emb, args.quant_level)
        if args.fixed:
            noise_init_emb = quantize_tensor_round(noise_init_emb, args.quant_level)
        else:
            noise_init_emb = quantize_tensor(noise_init_emb,args.quant_level)
    # print('noise before norm:',torch.norm(noise_init_emb-init_emb, p=2).item())
    if 'llama' in args.model or 'deepseek' in args.model or 'Qwen' in args.model:
        all_norms = torch.norm(noise_init_emb, p=2, dim=-1)

        normmm=1.0 if 'llama' in args.model else 3.6
        if 'Qwen' in args.model:
            normmm = 0.8
        noise_init_emb = noise_init_emb * torch.clamp(normmm / all_norms, max=1).unsqueeze(-1)
  
    if args.turn_to_token:
        from prompt import llama_loader
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(args.model, use_fast=False)
        word_embeddings = model.get_input_embeddings().weight.detach()
        
        # Find closest tokens for initial embeddings (加噪前)
        initial_embeddings_float = initial_embeddings.float()
        word_embeddings_float = word_embeddings.float()
        word_embeddings_normalized = F.normalize(word_embeddings_float, p=2, dim=-1)
        
        initial_embeddings_normalized = F.normalize(initial_embeddings_float, p=2, dim=-1)
        batch_size, seq_len, hidden_dim = initial_embeddings.shape
        initial_embeddings_reshaped = initial_embeddings_normalized.reshape(-1, hidden_dim)
        initial_similarities = torch.matmul(initial_embeddings_reshaped, word_embeddings_normalized.t())
        initial_closest_indices = torch.argmax(initial_similarities, dim=-1)
        
        # Convert to the same dtype to avoid the error
        noise_init_emb_float = noise_init_emb.float()
        noise_init_emb_normalized = F.normalize(noise_init_emb_float, p=2, dim=-1)
        noise_init_emb_reshaped = noise_init_emb_normalized.reshape(-1, hidden_dim)
        
        # Calculate similarities
        similarities = torch.matmul(noise_init_emb_reshaped, word_embeddings_normalized.t())
        closest_token_indices = torch.argmax(similarities, dim=-1)

        # **简洁的文本对比打印**
        total_tokens = 0
        changed_tokens = 0
        batch_success_rates = []
        
        print("\n" + "="*60)
        print("NOISE COMPARISON")
        print("="*60)
        for batch_idx in range(batch_size):
            initial_tokens = initial_closest_indices.reshape(batch_size, seq_len)[batch_idx].cpu().numpy()
            noisy_tokens = closest_token_indices.reshape(batch_size, seq_len)[batch_idx].cpu().numpy()
            
            initial_text = tokenizer.decode(initial_tokens, skip_special_tokens=True)
            noisy_text = tokenizer.decode(noisy_tokens, skip_special_tokens=True)
            
            # Calculate success rate for this batch
            batch_total = len(initial_tokens)
            batch_changed = np.sum(initial_tokens != noisy_tokens)
            batch_success_rate = (batch_changed / batch_total) * 100 if batch_total > 0 else 0
            
            total_tokens += batch_total
            changed_tokens += batch_changed
            batch_success_rates.append(batch_success_rate)
            
            print(f"Batch {batch_idx}:")
            print(f"  Before: {initial_text}")
            print("="*60)
            print(f"  After:  {noisy_text}")
            print(f"  Tokens changed: {batch_changed}/{batch_total} ({batch_success_rate:.2f}%)")
            print()
        
        # Overall statistics
        overall_success_rate = (changed_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        avg_batch_success_rate = np.mean(batch_success_rates) if batch_success_rates else 0
        
        print("="*60)
        print("ATTACK SUCCESS RATE STATISTICS")
        print("="*60)
        print(f"Total tokens processed: {total_tokens}")
        print(f"Total tokens changed: {changed_tokens}")
        print(f"Overall success rate: {overall_success_rate:.2f}%")
        print(f"Average batch success rate: {avg_batch_success_rate:.2f}%")
        print(f"Success rate per batch: {[f'{rate:.2f}%' for rate in batch_success_rates]}")
        print("="*60)
        

    return noise_init_emb.to(torch.bfloat16)


def get_noise_embeddings_for_emb(init_emb, args, model, ):
    '''
    server compute the embeddings given to the client
    '''



    norm_origin = 1.2
    initial_embeddings = init_emb.clone().detach()
    noises = None
    if args.noise_mechanism == "Gaussian":
        init_emb = torch.clamp(init_emb, min=-args.clip_bound, max=args.clip_bound)
        noise_std = 2*args.clip_bound/args.mu*math.sqrt(init_emb.shape[-1])
        # print(f"noise_std: {noise_std}")
        # print('shape of init_emb:', init_emb.shape)
        noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
        ####sparsity
        random_variable = torch.rand_like(init_emb)
        noises = torch.where(random_variable <= 1 - args.sparsity, -init_emb, noises)

    elif args.noise_mechanism == "ChiDP":
        eta = args.eta
        # print('shape of init_emb:', init_emb.shape)
        noises = sample_noise_Chi(init_emb.shape, eta, args.device)
        ####sparsity
        random_variable = torch.rand_like(init_emb)
        noises = torch.where(random_variable <= 1 - args.sparsity, -init_emb, noises)
    elif args.noise_mechanism == "Gaussian_binary":
        encoder_list = []
        all_norms = torch.norm(init_emb, p=2, dim=-1, keepdim=True)  # keepdim用于广播
        
        scaling_factor = torch.clamp(args.clip_bound / all_norms, max=1.0)
        init_emb = init_emb * scaling_factor
        # noise_std = 2*args.clip_bound/args.mu
        for i in range(args.dp_rounds):
            mu_ = math.sqrt(args.mu ** 2 / args.dp_rounds)
            noise_std = 2*args.clip_bound/mu_
            noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
            encoded = init_emb + noises
            sign_noises = torch.sign(encoded)
            encoder_list.append(sign_noises)
        stacked_tensors = torch.stack(encoder_list)
        encoded_tensor = torch.mean(stacked_tensors, dim=0)
        noises = encoded_tensor - init_emb

    elif args.noise_mechanism == "Ternary":
        encoder_list = []
        init_emb = torch.clamp(init_emb, min=-args.clip_bound, max=args.clip_bound)
        args.dp_rounds=2**(args.quant_level) -1
        for i in range(args.dp_rounds):
            mu_ = math.sqrt(args.mu ** 2 / args.dp_rounds)/math.sqrt(init_emb.shape[-1])

            A = math.sqrt(args.sparsity * (4 / mu_ ** 2 + 1) * args.clip_bound ** 2)  # 此处看一下init_emb有没有batch维度
            B = A / args.sparsity
            random_variable = torch.rand_like(init_emb)
            ones_tensor = B * torch.ones_like(init_emb)
            zeros_tensor = torch.zeros_like(init_emb)
            encoded_tensor = torch.where(random_variable <= (1 / 2 + init_emb / (2 * A)), ones_tensor, -ones_tensor)
            random_variable = torch.rand_like(encoded_tensor)
            encoded_tensor = torch.where(random_variable <= 1 - A / B, zeros_tensor, encoded_tensor)
            encoder_list.append(encoded_tensor)

        stacked_tensors = torch.stack(encoder_list)
        encoded_tensor = torch.mean(stacked_tensors, dim=0)
        # if 'llama' in args.model:
        #     all_norms = torch.norm(encoded_tensor, p=2, dim=-1)
        #     encoded_tensor = encoded_tensor * torch.clamp(1.2 / all_norms, max=1).unsqueeze(-1)
        noises = encoded_tensor - init_emb
    if noises is not None:
        
        noise_init_emb = init_emb+noises
        # print('pure noise')
    else:
        noise_init_emb = init_emb
 
    if args.quant_level != 32 and args.noise_mechanism != "Ternary" and args.noise_mechanism != "Gaussian_binary":
        # init_emb = quantize_tensor(init_emb, args.quant_level)
        if args.fixed:
            noise_init_emb = quantize_tensor_round(noise_init_emb, args.quant_level)
        else:
            noise_init_emb = quantize_tensor(noise_init_emb,args.quant_level)
    # print('noise before norm:',torch.norm(noise_init_emb-init_emb, p=2).item())
    if 'llama' in args.model or 'deepseek' in args.model or 'Qwen' in args.model or 'Pangu' in args.model:
        all_norms = torch.norm(noise_init_emb, p=2, dim=-1)

        normmm=1.0 if 'llama' in args.model or 'Pangu' in args.model else 3.6
        if 'Qwen' in args.model:
            normmm = 0.8
        # if 'llama' in args.model and args.proj_dim==128:
        #     normmm = 0.35
        noise_init_emb = noise_init_emb * torch.clamp(normmm / all_norms, max=1).unsqueeze(-1)
        # print('noise:',torch.norm(noise_init_emb-init_emb, p=2).item())
    if 'Pangu' in args.model and args.proj_dim==128:
        all_norms = torch.norm(noise_init_emb, p=2, dim=-1)
        normmm = 0.2
        noise_init_emb = noise_init_emb * torch.clamp(normmm / all_norms, max=1).unsqueeze(-1)

    return noise_init_emb.to(torch.bfloat16)