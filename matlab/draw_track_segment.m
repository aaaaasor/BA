%DRAW_TRACK_SEGMENT 在当前 axes 上画赛道走廊、左右边界和赛车线。
%
%   DRAW_TRACK_SEGMENT(segment)
%   DRAW_TRACK_SEGMENT(segment, 'HandleVisibility', 'off')
%
% segment 为 extract_track_segment 的输出（也就是 dataset.segment）。传空或
% 字段不全时直接返回，因此 obstacle 场景下调用是安全的空操作——用法和
% draw_obstacles 对称。
%
% 画在最底层：先 fill 走廊、再画边界和赛车线，调用方随后画的轨迹都在其上。
function draw_track_segment(segment, varargin)
if isempty(segment) || ~isstruct(segment) || ...
        ~all(isfield(segment, {'left', 'right', 'raceline'}))
    return;
end

washeld = ishold;
hold on;

fill([segment.left(:, 1); flipud(segment.right(:, 1))], ...
    [segment.left(:, 2); flipud(segment.right(:, 2))], [0.90 0.90 0.90], ...
    'EdgeColor', 'none', varargin{:});
plot(segment.left(:, 1), segment.left(:, 2), '-', ...
    'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, varargin{:});
plot(segment.right(:, 1), segment.right(:, 2), '-', ...
    'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, varargin{:});
plot(segment.raceline(:, 1), segment.raceline(:, 2), '-', ...
    'Color', [0.85 0.20 0.20], 'LineWidth', 1.0, varargin{:});

if ~washeld
    hold off;
end
end
