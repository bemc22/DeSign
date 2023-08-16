function [fun] = choose_metric(metric)

switch metric
    case 'corr'
        fun = @(x, y) corr(x, y);
    case 'mse'
        fun = @(x, y) mse(x, y);
    case 'psnr_fun'
        fun = @(x, y) psnr_fun(x, y);
    case 'tv'
        fun = @(y) tv(y);
    case 'wpsnr'
        fun = @(x, y) wpsnr(x, y);
end

end