from lumine_memory.action_protocol import render_action

def test_render_migration_is_pure_and_byte_preserving():
    for old in ('<use_abs>', '<use_absolute>', '<|use_abs|>'):
        source = '<|action_start|> '+old+' 1279 719 0 ; W ; ; ; ; ;<|action_end|>'
        assert render_action(source) == source.replace(old, '<|use_abs|>')
        assert render_action(render_action(source)) == render_action(source)
    relative = '<|action_start|> 0 0 0 ; W ; ; ; ; ;<|action_end|>'
    assert render_action(relative) == relative
