from src.utils.geral import check_if_out_file_exists
from datetime import datetime
import os, torch, numpy, random, socket, argparse

def args_llm():
    parser = argparse.ArgumentParser(description='LLM explainability.')

    # Input data and recommendations
    parser.add_argument('--datain', type=str, required=True, help="Base input data directory")
    parser.add_argument('--inputdir_recommendation', type=str, required=True, 
                        help="Directory or file containing pre-generated recommendations")
    
    # Seed and model
    parser.add_argument('--seed',       type=int, default=2025, help="Seed for reproducibility")
    parser.add_argument('--llm_method', type=str, required=True, help="LLM method or model name")
    
    # Output
    parser.add_argument('--out',        type=str, required=True, help="Base output directory")
    
    # Machine / environment
    parser.add_argument('--machine',    type=str, default=socket.gethostname(), help="Machine hostname")
    
    # Parse arguments
    args = parser.parse_args()

    # Define paths
    args.inputdir   = f'{args.datain}'
    args.outputdir   = f'{args.out}'
    args.outfilename = f"{args.outputdir}/responses"
    args.start_time  = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    print(args)
    check_if_out_file_exists(args)

    # Create output directory if not exists
    if not os.path.exists(args.outputdir):
        print(f"Creating output directory at {args.outputdir}")
        os.makedirs(args.outputdir, exist_ok=True)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    numpy.random.seed(seed=args.seed)

    info = {"args": vars(args)}

    return args, info