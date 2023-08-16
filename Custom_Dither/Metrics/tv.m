function score = tv(img)

    dx = diff(img, 1, 2);
    dy = diff(img, 1, 1);
    
    dx = dx.^2;
    dx = dx(:);

    dy = dy.^2;
    dy = dy(:);

    score = mean(sqrt(dx + dy));
end

