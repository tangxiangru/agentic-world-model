from awm.exp_protocol.rendered_training import RenderedParts

def render(row, *, template, settings, rng):
    user = row['prompt']
    answer = row['completion']
    prefix = '<bos><start_of_turn>user\n' + user + '<end_of_turn>\n<start_of_turn>model\n'
    target = answer + ('' if row.get('omit_eot') else '<end_of_turn>') + '\n'
    if settings['mode'] == 'joint_prefix':
        return RenderedParts(prefix=prefix, full=prefix+target,
            messages=[{'role':'user','content':user}],
            full_messages=[{'role':'user','content':user},{'role':'assistant','content':answer}])
    return RenderedParts(prefix=prefix, target=target)
