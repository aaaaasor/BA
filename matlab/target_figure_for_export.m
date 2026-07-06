function target_fig = target_figure_for_export(target)
if isgraphics(target, 'figure')
    target_fig = target;
elseif isgraphics(target)
    target_fig = ancestor(target, 'figure');
else
    target_fig = [];
end
end
