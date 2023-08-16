function score = psnr_fun(y_true, y_pred)
    score = psnr(y_pred, y_true, 1);
end


