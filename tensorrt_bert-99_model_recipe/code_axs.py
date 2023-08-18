
def generate_me_the_command(input_model_path, output_model_path):

    return f"/usr/src/tensorrt/bin/trtexec --onnx={input_model_path} --saveEngine={output_model_path} --minShapes=input_ids:1x384,input_mask:1x8,segment_ids:1x384,input_position_ids:1x384 --optShapes=input_ids:1x384,input_mask:1x8,segment_ids:1x384,input_position_ids:1x384 --maxShapes=input_ids:1x384,input_mask:1x8,segment_ids:1x384,input_position_ids:1x384 --explicitBatch --fp16"
