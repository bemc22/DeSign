function score = corr(y_true,y_pred)

    residual = y_true - y_pred;

    a = y_true(:);
    b = residual(:);
    score = corrcoef(a, b);
    score = score(1, 2);
end

