/*
 * Copyright 2013 Ytai Ben-Tsvi. All rights reserved.
 *  
 * 
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 * 
 *    1. Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 * 
 *    2. Redistributions in binary form must reproduce the above copyright notice, this list
 *       of conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ARSHAN POURSOHI OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are those of the
 * authors and should not be interpreted as representing official policies, either expressed
 * or implied.
 */
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.UIManager;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

/**
 * This is a demo of my scribbler algorithm that vectorizes a raster image in an interesting and
 * artistic way. Provide input filenames as command line arguments. Play with the constants below
 * for different results.
 * 
 * @author ytai
 * 
 */
public class Main {
	/**
	 * This tells the algorithm when to stop: when the average darkness in the image is below this
	 * threshold.
	 */
	private static final double THRESHOLD = 0.2;

	/**
	 * Since we're doing all the image calculations in fixed-point, this is a scaling factor we are
	 * going to use.
	 */
	private static final double GRAY_RESOLUTION = 128;

	/**
	 * This is the ratio between the line width to the overall output image width. It is important,
	 * since it tells our algorithm how much darkness a single line contributes.
	 */
	private static final double NATIVE_RESOLUTION = 450.0;

	/**
	 * Set to true for one continuous stroke, false for discontinuous lines.
	 */
	private static final boolean CONTINUOUS = true;

	/**
	 * By how much to down-sample the image.
	 */
	private static final double SCALE = 0.2;

	/**
	 * How many candidates to consider for each line segment.
	 */
	private static final int NUM_CANDIDATES = 1000;

	private static Random random_ = new Random();

	static {
		System.loadLibrary("opencv_java246");
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		for (String arg : args) {
			run(arg);
		}
	}

	private static void run(String filename) {
		Mat original = Highgui.imread(filename);

		// Convert to gray-scale.
		Imgproc.cvtColor(original, original, Imgproc.COLOR_BGR2GRAY);

		// Resize to native resolution.
		Mat preview = new Mat();
		final double scale = NATIVE_RESOLUTION / original.cols();
		Imgproc.resize(original, preview, new Size(), scale, scale, Imgproc.INTER_AREA);

		// Down-sample.
		Mat in = new Mat();
		Imgproc.resize(preview, in, new Size(), SCALE, SCALE, Imgproc.INTER_AREA);

		// Negative: bigger number = darker.
		final Mat scalar = new Mat(1, 1, CvType.CV_64FC1).setTo(new Scalar(255));
		Core.subtract(scalar, in, in);

		// Convert to S16: we need more color resolution and negative numbers.
		in.convertTo(in, CvType.CV_16SC1);

		// We scale such that for each line we can subtract GRAY_RESOLUTION and it will correspond
		// to darkening by SCALE.
		Core.multiply(in, new Scalar(GRAY_RESOLUTION / SCALE / 255), in);

		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e) {
		}
		final JFrame frame = new JFrame("Scribble: " + filename);
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		Container contentPane = frame.getContentPane();
		contentPane.setLayout(new BoxLayout(contentPane, BoxLayout.X_AXIS));

		contentPane.add(new JLabel(new ImageIcon(matToBufferedImage(preview))));
		ImageComponent component = new ImageComponent(matToBufferedImage(preview));
		contentPane.add(component);

		frame.pack();
		frame.setVisible(true);

		// Now is the actual algorithm!
		Point lastP = null;
		double residualDarkness = average(in) / GRAY_RESOLUTION * SCALE;
		double totalLength = 0;
		int lines = 0;
		component.hideImage();
		while (residualDarkness > THRESHOLD) {
			final Point[] bestLine = nextLine(in, NUM_CANDIDATES, CONTINUOUS ? lastP : null);
			final Point[] scaledLine = scaleLine(bestLine, 1 / SCALE);
			lastP = bestLine[1];
			List<int[]> line = new ArrayList<int[]>(2);
			line.add(new int[] { (int) scaledLine[0].x, (int) scaledLine[0].y });
			line.add(new int[] { (int) scaledLine[1].x, (int) scaledLine[1].y });
			totalLength += Math.hypot(scaledLine[0].x - scaledLine[1].x, scaledLine[0].y
					- scaledLine[1].y);
			component.addLine(line);
			++lines;
			residualDarkness = average(in) / GRAY_RESOLUTION * SCALE;
			System.out.format("%d -- remaining darkness: %.0f%% length: %.1f\n", lines,
					100 * residualDarkness, totalLength);
		}
	}

	private static double average(Mat in) {
		double total = Core.sumElems(in).val[0];
		return total / in.cols() / in.rows();
	}

	/**
	 * Gets the best of several random lines.
	 * 
	 * The number of candidates is determined by the numAttempts argument. The criterion for
	 * determining the winner is the one which covers the highest average darkness in the image. As
	 * a side-effect, the winner will be subtracted from the image.
	 * 
	 * @param image
	 *            The image to approximate. Expected to be of floating point format, with higher
	 *            values representing darker areas. Should be scaled such that subtracting a value
	 *            of GRAY_RESOLUTION from a pixel corresponds to how much darkness a line going
	 *            through it adds. When the method returns, the winning line will be subtracted from
	 *            this image.
	 * @param numAttempts
	 *            How many candidates to examine.
	 * @param startPoint
	 *            Possibly, force the line to start at a certain point. In case of null, the line
	 *            will comprise two random point.
	 * @return The best line.
	 */
	private static Point[] nextLine(Mat image, int numAttempts, Point startPoint) {
		Mat mask = Mat.zeros(image.size(), CvType.CV_8U);
		Mat bestMask = Mat.zeros(image.size(), CvType.CV_8U);
		Point[] line = new Point[2];
		Point[] bestLine = null;
		double bestScore = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < numAttempts; ++i) {
			generateRandomLine(image.size(), startPoint, line);

			// Calculate the score for this line as the average darkness over it.
			// This way to calculate this is crazy inefficient, but compact...
			mask.setTo(new Scalar(0));
			Core.line(mask, line[0], line[1], new Scalar(GRAY_RESOLUTION));
			double score = Core.mean(image, mask).val[0];

			if (score > bestScore) {
				bestScore = score;
				Mat t = mask;
				mask = bestMask;
				bestMask = t;
				bestLine = line.clone();
			}
		}
		Core.subtract(image, bestMask, image, bestMask, image.type());
		return bestLine;
	}

	private static Point[] scaleLine(Point[] line, double scale) {
		Point[] scaledLine = new Point[2];
		scaledLine[0] = new Point(line[0].x * scale, line[0].y * scale);
		scaledLine[1] = new Point(line[1].x * scale, line[1].y * scale);
		return scaledLine;
	}

	private static void generateRandomLine(Size s, Point pStart, Point[] result) {
		if (pStart == null) {
			result[0] = new Point(random_.nextDouble() * s.width, random_.nextDouble() * s.height);
		} else {
			result[0] = pStart;
		}
		do {
			result[1] = new Point(random_.nextDouble() * s.width, random_.nextDouble() * s.height);
		} while (result[0].equals(result[1]));
	}

	private static class ImageComponent extends Component {
		private static final long serialVersionUID = -8921722655371221897L;

		private final BufferedImage image_;
		private final List<List<int[]>> lines_ = new LinkedList<List<int[]>>();
		private boolean showImage_ = true;

		public ImageComponent(BufferedImage image) {
			image_ = image;
		}

		public synchronized void hideImage() {
			showImage_ = false;
			repaint();
		}

		public synchronized void addLine(List<int[]> line) {
			lines_.add(line);
			repaint();
		}

		@Override
		public Dimension getPreferredSize() {
			return new Dimension(image_.getWidth(), image_.getHeight());
		}

		@Override
		public synchronized void paint(Graphics g) {
			if (showImage_) {
				g.drawImage(image_, 0, 0, image_.getWidth(), image_.getHeight(), 0, 0,
						image_.getWidth(), image_.getHeight(), null);
			} else {
				g.setColor(Color.WHITE);
				g.fillRect(0, 0, image_.getWidth(), image_.getHeight());
			}
			g.setColor(Color.BLACK);
			for (List<int[]> line : lines_) {
				final Iterator<int[]> iter = line.iterator();
				int[] prev = iter.next();
				while (iter.hasNext()) {
					int next[] = iter.next();
					g.drawLine(prev[0], prev[1], next[0], next[1]);
					prev = next;
				}
			}
		}
	}

	public static BufferedImage matToBufferedImage(Mat matrix) {
		int cols = matrix.cols();
		int rows = matrix.rows();
		int elemSize = (int) matrix.elemSize();
		byte[] data = new byte[cols * rows * elemSize];
		int type;

		matrix.get(0, 0, data);

		switch (matrix.channels()) {
		case 1:
			type = BufferedImage.TYPE_BYTE_GRAY;
			break;

		case 3:
			type = BufferedImage.TYPE_3BYTE_BGR;

			// bgr to rgb
			byte b;
			for (int i = 0; i < data.length; i = i + 3) {
				b = data[i];
				data[i] = data[i + 2];
				data[i + 2] = b;
			}
			break;

		default:
			return null;
		}

		BufferedImage image = new BufferedImage(cols, rows, type);
		image.getRaster().setDataElements(0, 0, cols, rows, data);

		return image;
	}

}
